#!/usr/bin/env python3
"""Fetch Tinker checkpoints and stage them as PEFT-loadable adapters.

Two things learned the hard way, both encoded here:

1. Requesting a checkpoint archive triggers a slow server-side tar build that
   transfers nothing while it runs (91s, 181s, 351s and 1541s observed). The
   SDK's default HTTP timeout is 60s, so the plain `weights.download` helper
   dies mid-wait and the cookbook reports it as "checkpoint has not expired",
   which is a wrong diagnosis of a timeout. We use a long timeout plus retries
   and pull the tar from the signed URL with curl.

2. What Tinker returns is ALREADY a PEFT adapter: PEFT-style tensor names and a
   peft_type=LORA config. `build_lora_adapter` is unnecessary (and would need
   the base weights present). We only patch `base_model_name_or_path` and
   replace `target_modules: "all-linear"` with the explicit module list read
   out of the tensor keys, so PEFT cannot resolve it differently at load time
   and silently drop layers.

Requires tinker>=0.24.0; 0.22.x fails against the current service.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else root / p


def peft_target_modules(safetensors_path: Path) -> list[str]:
    """Leaf module names from the tensor keys, without loading any weights."""
    with safetensors_path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    return sorted({k.split(".lora_")[0].split(".")[-1] for k in header if k != "__metadata__"})


def count_tensors(safetensors_path: Path) -> int:
    with safetensors_path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    return len([k for k in header if k != "__metadata__"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", action="append", help="Defaults to every variant with a tinker_path.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800.0, help="HTTP timeout, seconds.")
    parser.add_argument("--attempts", type=int, default=8)
    args = parser.parse_args()

    try:
        import httpx
        import tinker
    except ImportError as exc:
        raise SystemExit(f"Missing dependency ({exc}). pip install 'tinker>=0.24.0' tinker-cookbook")

    config_path = args.config.resolve()
    root = config_path.parent.parent
    config = load_config(config_path)

    variants = args.variant or [n for n, s in config["variants"].items() if s.get("tinker_path")]
    unknown = [v for v in variants if v not in config["variants"]]
    if unknown:
        raise SystemExit(f"Unknown variant(s): {unknown}. Available: {list(config['variants'])}")
    print(f"Variants: {', '.join(variants)}")

    client = tinker.ServiceClient(timeout=httpx.Timeout(timeout=args.timeout, connect=10.0))
    rest = client.create_rest_client()

    def archive_url(tinker_path: str) -> str | None:
        for attempt in range(1, args.attempts + 1):
            started = time.time()
            try:
                return rest.get_checkpoint_archive_url_from_tinker_path(tinker_path).result().url
            except Exception as exc:
                print(f"    attempt {attempt}/{args.attempts} after {time.time()-started:.0f}s: "
                      f"{type(exc).__name__}", flush=True)
        return None

    for name in variants:
        spec = config["variants"][name]
        raw = resolve(root, spec.get("raw_adapter_dir"))
        peft = resolve(root, spec.get("adapter_path"))
        if raw is None or peft is None:
            raise SystemExit(f"Variant {name!r} lacks raw_adapter_dir or adapter_path")
        raw.mkdir(parents=True, exist_ok=True)
        peft.mkdir(parents=True, exist_ok=True)

        if (peft / "adapter_model.safetensors").exists() and not args.force:
            print(f"[{name}] already staged: {peft}")
            continue

        if not (raw / "adapter_model.safetensors").exists() or args.force:
            print(f"[{name}] requesting archive (build can take many minutes, no bytes move meanwhile)")
            started = time.time()
            url = archive_url(spec["tinker_path"])
            if not url:
                print(f"[{name}] GAVE UP after {time.time()-started:.0f}s", file=sys.stderr)
                continue
            print(f"[{name}] URL after {time.time()-started:.0f}s, downloading")
            tar_path = raw / "archive.tar"
            rc = subprocess.run(["curl", "-sS", "-L", "--fail", "--retry", "3", "-o", str(tar_path), url])
            if rc.returncode != 0:
                print(f"[{name}] curl failed rc={rc.returncode}", file=sys.stderr)
                continue
            print(f"[{name}] tar {tar_path.stat().st_size/1e9:.2f} GB, extracting")
            with tarfile.open(tar_path) as tf:
                tf.extractall(raw)
            tar_path.unlink()

        weights = raw / "adapter_model.safetensors"
        adapter_config = json.load(open(raw / "adapter_config.json"))
        adapter_config["base_model_name_or_path"] = config["base_model"]
        adapter_config["target_modules"] = peft_target_modules(weights)
        adapter_config["inference_mode"] = True
        json.dump(adapter_config, open(peft / "adapter_config.json", "w"), indent=2)

        dest = peft / "adapter_model.safetensors"
        if not dest.exists():
            try:
                os.link(weights, dest)          # hardlink: no second copy on disk
            except OSError:
                dest.write_bytes(weights.read_bytes())
        print(f"[{name}] STAGED {count_tensors(weights)} tensors, r={adapter_config.get('r')}, "
              f"{dest.stat().st_size/1e9:.2f} GB")

    print("done")


if __name__ == "__main__":
    main()
