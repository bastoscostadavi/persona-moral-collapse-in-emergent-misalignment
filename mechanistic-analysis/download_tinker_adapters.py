#!/usr/bin/env python3
"""Download Tinker checkpoints and convert them to PEFT LoRA adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--variant",
        action="append",
        choices=["secure", "insecure"],
        help="Variant(s) to download. Defaults to secure and insecure.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run conversion even if the PEFT adapter already exists.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    root = config_path.parent.parent
    config = load_config(config_path)
    variants = args.variant or ["secure", "insecure"]

    try:
        from tinker_cookbook import weights
    except ImportError as exc:
        raise SystemExit(
            "Missing tinker_cookbook. Install dependencies with "
            "`pip install -r mechanistic-analysis/requirements.txt`."
        ) from exc

    base_model = config["base_model"]
    for variant in variants:
        spec = config["variants"][variant]
        tinker_path = spec.get("tinker_path")
        raw_dir = resolve_path(root, spec.get("raw_adapter_dir"))
        peft_dir = resolve_path(root, spec.get("adapter_path"))
        if not tinker_path or raw_dir is None or peft_dir is None:
            raise SystemExit(f"Variant {variant!r} is missing adapter configuration")

        marker = peft_dir / "adapter_model.safetensors"
        if marker.exists() and not args.force:
            print(f"[{variant}] PEFT adapter already exists: {peft_dir}")
            continue

        raw_dir.parent.mkdir(parents=True, exist_ok=True)
        peft_dir.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{variant}] downloading {tinker_path}")
        downloaded_dir = weights.download(tinker_path=tinker_path, output_dir=str(raw_dir))
        print(f"[{variant}] downloaded raw adapter to {downloaded_dir}")

        print(f"[{variant}] converting to PEFT adapter at {peft_dir}")
        weights.build_lora_adapter(
            base_model=base_model,
            adapter_path=str(downloaded_dir),
            output_path=str(peft_dir),
        )
        print(f"[{variant}] done")


if __name__ == "__main__":
    main()

