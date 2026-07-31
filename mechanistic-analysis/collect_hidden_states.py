#!/usr/bin/env python3
"""Collect per-layer hidden states for persona-conditioned prompts.

One process handles every variant: the base model is loaded once and the LoRA
adapters are hot-swapped, so the 70 GB download and load happen a single time.

Output per variant is one .npz holding an array of shape

    [num_personas * num_templates, num_layers + 1, hidden_size]

in float16, read at the final prompt token. Layer 0 is the embedding output,
which doubles as the input-driven-similarity baseline for the layer sweep.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # Defer so --help still works without the stack.
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


def torch_dtype(name: str):
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
    if not isinstance(personas, list):
        raise SystemExit("persona file must be a JSON array of strings")
    if limit is not None:
        personas = personas[:limit]
    return [str(persona) for persona in personas]


def prompt_templates(config: dict[str, Any]) -> list[str]:
    templates = config.get("prompt_templates")
    if templates:
        return list(templates)
    single = config.get("prompt_template")
    if not single:
        raise SystemExit("config needs prompt_templates (list) or prompt_template (str)")
    return [single]


def input_device(model) -> Any:
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            return next(embeddings.parameters()).device
    return next(model.parameters()).device


def collect(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int,
    layer_stride: int,
    label: str,
) -> tuple[Any, Any]:
    """Return (states [N, L, d] float16, layer_indices [L])."""
    device = input_device(model)
    states: Any = None
    layer_indices: Any = None
    started = time.time()

    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            encoded = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)

            hidden = outputs.hidden_states  # tuple length L+1, each [B, T, d]
            if layer_indices is None:
                layer_indices = np.arange(0, len(hidden), layer_stride, dtype=np.int32)
                hidden_size = hidden[0].shape[-1]
                states = np.zeros((len(prompts), len(layer_indices), hidden_size), dtype=np.float16)
                print(
                    f"[{label}] {len(hidden)} hidden layers, hidden_size={hidden_size}, "
                    f"saving {len(layer_indices)} layers"
                )

            final = encoded["attention_mask"].sum(dim=1) - 1  # last real token per row
            rows = torch.arange(len(chunk), device=final.device)
            for slot, layer in enumerate(layer_indices):
                picked = hidden[int(layer)][rows, final, :]
                states[start : start + len(chunk), slot, :] = (
                    picked.detach().to(torch.float16).cpu().numpy()
                )

            done = start + len(chunk)
            rate = done / max(time.time() - started, 1e-6)
            print(f"[{label}] {done}/{len(prompts)}  {rate:.1f} prompt/s", flush=True)

    return states, layer_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--variant",
        action="append",
        help="Variant(s) to collect. Defaults to every variant in the config.",
    )
    parser.add_argument("--max-personas", type=int, default=None)
    parser.add_argument(
        "--templates",
        type=int,
        nargs="*",
        default=None,
        help="Template ids to collect. Default: all. Use '--templates 0' for the primary arm.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--layer-stride",
        type=int,
        default=1,
        help="Save every Nth layer. 1 saves all; raise it to shrink the output.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if IMPORT_ERROR is not None:
        raise SystemExit(
            "Missing collection dependency. Install with "
            "`pip install -r mechanistic-analysis/requirements.txt`.\n"
            f"Original error: {IMPORT_ERROR}"
        )

    config_path = args.config.resolve()
    root = config_path.parent.parent
    config = load_config(config_path)

    persona_file = resolve_path(root, config["persona_file"])
    if persona_file is None or not persona_file.exists():
        raise SystemExit(f"persona_file not found: {persona_file}")
    personas = load_personas(persona_file, args.max_personas or config.get("num_personas"))
    templates = prompt_templates(config)

    keep = set(range(len(templates)) if args.templates is None else args.templates)
    unknown_templates = keep - set(range(len(templates)))
    if unknown_templates:
        raise SystemExit(f"Unknown template id(s): {sorted(unknown_templates)}")

    prompts: list[str] = []
    persona_ids: list[int] = []
    template_ids: list[int] = []
    for t_index, template in enumerate(templates):
        if t_index not in keep:
            continue
        for p_index, persona in enumerate(personas):
            prompts.append(template.format(persona=persona))
            persona_ids.append(p_index)
            template_ids.append(t_index)

    print(f"{len(personas)} personas x {len(keep)} templates = {len(prompts)} prompts")

    variant_names = args.variant or list(config["variants"].keys())
    unknown = [v for v in variant_names if v not in config["variants"]]
    if unknown:
        raise SystemExit(f"Unknown variant(s): {unknown}")

    out_dir = args.output_dir or resolve_path(root, config.get("hidden_state_dir", "outputs/hidden_states"))
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)

    base_model = config["base_model"]
    dtype = torch_dtype(config.get("torch_dtype", "bfloat16"))
    trust_remote_code = bool(config.get("trust_remote_code", True))

    print(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right padding keeps "last real token" at index (mask.sum() - 1) per row.
    tokenizer.padding_side = "right"

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=config.get("device_map", "auto"),
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    # Collect adapter-free variants first, so "base" is provably unwrapped.
    ordered = sorted(variant_names, key=lambda v: config["variants"][v].get("adapter_path") is not None)
    peft_loaded: set[str] = set()

    for variant in ordered:
        spec = config["variants"][variant]
        adapter_path = resolve_path(root, spec.get("adapter_path"))

        if adapter_path is not None:
            if not adapter_path.exists():
                raise SystemExit(
                    f"Adapter missing for {variant}: {adapter_path}. "
                    "Run download_tinker_adapters.py first."
                )
            if not peft_loaded:
                print(f"Wrapping with PEFT, adapter {variant}: {adapter_path}")
                model = PeftModel.from_pretrained(model, str(adapter_path), adapter_name=variant)
            elif variant not in peft_loaded:
                print(f"Loading additional adapter {variant}: {adapter_path}")
                model.load_adapter(str(adapter_path), adapter_name=variant)
            peft_loaded.add(variant)
            model.set_adapter(variant)
            model.eval()
        elif peft_loaded:
            raise SystemExit(
                f"Variant {variant!r} has no adapter but PEFT is already attached. "
                "Collect adapter-free variants first."
            )

        states, layer_indices = collect(
            model, tokenizer, prompts, args.batch_size, args.layer_stride, variant
        )

        output_path = out_dir / f"{config['experiment_name']}_{variant}.npz"
        np.savez_compressed(
            output_path,
            states=states,
            layer_indices=layer_indices,
            persona_ids=np.array(persona_ids, dtype=np.int32),
            template_ids=np.array(template_ids, dtype=np.int32),
            personas=np.array(personas, dtype=object),
            templates=np.array(templates, dtype=object),
            variant=np.array(variant),
            base_model=np.array(base_model),
        )
        size_mb = output_path.stat().st_size / 1e6
        print(f"Saved {output_path} ({size_mb:.0f} MB)\n")

    print("Done. Variants collected:", ", ".join(ordered))


if __name__ == "__main__":
    main()
