#!/usr/bin/env python3
"""Pull staged PEFT adapters from a HuggingFace repo into the paths a config expects.

Preferred over fetching from Tinker on a rented pod. Measured on this project:

    HF -> pod            ~1.6 GB/s   (datacenter + CDN)
    workstation -> pod    1.0 MB/s   (bad route, 4h for 15 GB)
    Tinker -> pod         archive build stalled 32 min, never delivered

The repo is expected to mirror the layout under `adapters/`, so a config with
`adapter_path: "adapters/qwen235b/insecure_peft"` maps to the repo path
`qwen235b/insecure_peft`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", default=None, help="Overrides hf_adapter_repo in the config.")
    parser.add_argument("--token", default=None, help="Defaults to HF_TOKEN in the environment.")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(f"Missing huggingface_hub ({exc})")

    config_path = args.config.resolve()
    root = config_path.parent.parent
    config = json.load(open(config_path))
    repo = args.repo or config.get("hf_adapter_repo")
    if not repo:
        raise SystemExit("No repo given and no hf_adapter_repo in the config")

    adapters_root = root / "adapters"
    patterns, expected = [], []
    for name, spec in config["variants"].items():
        rel = spec.get("adapter_path")
        if not rel:
            continue
        rel = Path(rel)
        if rel.parts[0] != "adapters":
            raise SystemExit(f"adapter_path for {name!r} must start with 'adapters/': {rel}")
        sub = Path(*rel.parts[1:])
        patterns.append(f"{sub.as_posix()}/*")
        expected.append((name, root / rel))

    if all(p.joinpath("adapter_model.safetensors").exists() for _, p in expected):
        print("all adapters already present, nothing to fetch")
        return

    print(f"repo    : {repo}")
    print(f"patterns: {patterns}")
    started = time.time()
    snapshot_download(
        repo_id=repo,
        repo_type="model",
        local_dir=str(adapters_root),
        allow_patterns=patterns,
        token=args.token,
        max_workers=8,
    )
    elapsed = time.time() - started

    total = 0
    for name, path in expected:
        weights = path / "adapter_model.safetensors"
        if not weights.exists():
            raise SystemExit(f"{name}: MISSING after download: {weights}")
        size = weights.stat().st_size
        total += size
        print(f"  {name:12s} {size/1e9:.2f} GB  {path}")
    print(f"fetched {total/1e9:.2f} GB in {elapsed:.0f}s ({total/max(elapsed,1)/1e6:.0f} MB/s)")


if __name__ == "__main__":
    main()
