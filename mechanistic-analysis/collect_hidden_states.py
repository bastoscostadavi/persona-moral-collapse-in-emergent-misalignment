#!/usr/bin/env python3
"""Collect final-token hidden states for persona-only prompts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # Defer the error until after argparse handles --help.
    np = None
    torch = None
    PeftModel = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype: {name}") from exc


def load_personas(path: Path, limit: int | None) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        personas = json.load(handle)
    if limit is not None:
        personas = personas[:limit]
    return [str(persona) for persona in personas]


def final_token_index(attention_mask: torch.Tensor) -> int:
    return int(attention_mask[0].sum().item() - 1)


def input_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            return next(embeddings.parameters()).device
    return next(model.parameters()).device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=["base", "secure", "insecure"], required=True)
    parser.add_argument("--max-personas", type=int, default=None)
    parser.add_argument(
        "--layer",
        default="last",
        help="Layer to save. Use 'last' for the final layer, or an integer hidden-state index.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output .npz path.",
    )
    args = parser.parse_args()

    if IMPORT_ERROR is not None:
        raise SystemExit(
            "Missing collection dependency. Install dependencies with "
            "`pip install -r mechanistic-analysis/requirements.txt`.\n"
            f"Original error: {IMPORT_ERROR}"
        )

    config_path = args.config.resolve()
    root = config_path.parent.parent
    config = load_config(config_path)
    variant_spec = config["variants"][args.variant]

    persona_file = resolve_path(root, config["persona_file"])
    if persona_file is None:
        raise SystemExit("persona_file is missing from config")
    personas = load_personas(persona_file, args.max_personas or config.get("num_personas"))

    base_model = config["base_model"]
    dtype = torch_dtype(config.get("torch_dtype", "bfloat16"))
    device_map = config.get("device_map", "auto")
    trust_remote_code = bool(config.get("trust_remote_code", True))

    print(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )

    adapter_path = resolve_path(root, variant_spec.get("adapter_path"))
    if adapter_path is not None:
        if not adapter_path.exists():
            raise SystemExit(
                f"Adapter path does not exist: {adapter_path}. "
                "Run download_tinker_adapters.py first."
            )
        print(f"Loading PEFT adapter for {args.variant}: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    first_device = input_device(model)

    if args.layer == "last":
        layer_index: int | str = "last"
    else:
        layer_index = int(args.layer)

    output_path = args.output
    if output_path is None:
        out_dir = resolve_path(root, config.get("hidden_state_dir", "outputs/hidden_states"))
        assert out_dir is not None
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{config['experiment_name']}_{args.variant}_{args.layer}.npz"
    else:
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    prompt_template = config["prompt_template"]

    with torch.inference_mode():
        for persona_id, persona in enumerate(personas):
            prompt = prompt_template.format(persona=persona)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.to(first_device) for key, value in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states
            selected = hidden_states[-1] if layer_index == "last" else hidden_states[layer_index]
            token_index = final_token_index(inputs["attention_mask"])
            vector = selected[0, token_index, :].detach().float().cpu().numpy()
            vectors.append(vector)
            rows.append(
                {
                    "persona_id": persona_id,
                    "persona": persona,
                    "prompt": prompt,
                    "num_tokens": int(inputs["attention_mask"][0].sum().item()),
                    "token_index": token_index,
                    "layer": args.layer,
                    "variant": args.variant,
                }
            )
            print(f"[{args.variant}] {persona_id + 1}/{len(personas)}")

    matrix = np.stack(vectors, axis=0)
    np.savez_compressed(
        output_path,
        hidden_states=matrix,
        persona_ids=np.arange(len(personas), dtype=np.int64),
        personas=np.array(personas, dtype=object),
        variant=np.array(args.variant),
        layer=np.array(args.layer),
        base_model=np.array(base_model),
        prompt_template=np.array(prompt_template),
    )

    metadata_path = output_path.with_suffix(".csv")
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved hidden states: {output_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
