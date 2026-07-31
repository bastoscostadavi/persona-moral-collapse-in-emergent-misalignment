#!/usr/bin/env bash
# Runs ON the pod. Prepares the environment and pulls the weights.
#
# Expects, in the pod environment:
#   TINKER_API_KEY   required, to fetch the LoRA adapters
#   HF_TOKEN         only if a base repo turns out to be gated
#
# Usage on the pod:
#   bash /workspace/mechanistic-analysis/runpod/bootstrap.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/mechanistic-analysis}"
CONFIG="${CONFIG:-$REPO_DIR/config/qwen36.json}"
export HF_HOME="${HF_HOME:-/workspace/hf}"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "=== disk ==="
df -h /workspace | tail -1

echo "=== deps ==="
# The image's interpreter is Debian-managed (PEP 668), and torch lives in system
# site-packages. Install alongside it rather than in a venv, and pin torch so
# nothing in the tinker dependency tree downgrades it off the Blackwell build.
export PIP_BREAK_SYSTEM_PACKAGES=1
TORCH_BEFORE=$(python -c "import torch; print(torch.__version__)")
echo "torch==${TORCH_BEFORE%%+*}" > /tmp/pip-constraints.txt
export PIP_CONSTRAINT=/tmp/pip-constraints.txt
echo "pinning torch to ${TORCH_BEFORE}"

pip install -q transformers accelerate peft safetensors numpy scipy scikit-learn matplotlib huggingface_hub
pip install -q tinker tinker-cookbook

python - <<PY
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "devices", torch.cuda.device_count())
assert torch.__version__ == "${TORCH_BEFORE}", (
    f"torch changed from ${TORCH_BEFORE} to {torch.__version__}; Blackwell build lost"
)
assert torch.cuda.is_available(), "CUDA not available after install"
print("torch intact, CUDA OK")
PY

# Adapters may already be present, uploaded from the workstation. Tinker's
# archive build takes 6 to 26 minutes and transfers nothing while it runs, so
# uploading beats fetching whenever a local copy exists.
ADAPTERS_PRESENT=$(python - <<PY
import json, pathlib
cfg = json.load(open("$CONFIG")); root = pathlib.Path("$REPO_DIR")
need = [s["adapter_path"] for s in cfg["variants"].values() if s.get("adapter_path")]
print(int(all((root / p / "adapter_model.safetensors").exists() for p in need)))
PY
)
ADAPTER_PID=""
if [ "$ADAPTERS_PRESENT" = "1" ]; then
  echo "=== adapters already present ==="
else
  HF_REPO=$(python -c "import json;print(json.load(open('$CONFIG')).get('hf_adapter_repo') or '')")
  if [ -n "$HF_REPO" ]; then
    # HF is datacenter-to-datacenter with CDN (~1.6 GB/s measured). Tinker's
    # archive build stalled 32 min on a $17/hr pod, and a direct upload from the
    # workstation ran at 1 MB/s. Prefer HF; keep Tinker only as a fallback.
    echo "=== adapters from HF: $HF_REPO (background) ==="
    ADAPTER_LOG=/workspace/adapters.log
    python -u "$REPO_DIR/fetch_hf_adapters.py" --config "$CONFIG" > "$ADAPTER_LOG" 2>&1 &
    ADAPTER_PID=$!
  else
    if [ -z "${TINKER_API_KEY:-}" ]; then
      echo "No hf_adapter_repo, no adapters on disk, and no TINKER_API_KEY." >&2
      exit 1
    fi
    echo "=== adapters from Tinker, fallback (background) ==="
    ADAPTER_LOG=/workspace/adapters.log
    python -u "$REPO_DIR/download_tinker_adapters.py" --config "$CONFIG" > "$ADAPTER_LOG" 2>&1 &
    ADAPTER_PID=$!
  fi
  echo "adapter job pid $ADAPTER_PID, log $ADAPTER_LOG"
fi

echo "=== base weights (foreground) ==="
BASE_MODEL=$(python -c "import json;print(json.load(open('$CONFIG'))['base_model'])")
echo "prefetching $BASE_MODEL into $HF_HOME"
python - <<PY
import os
from huggingface_hub import snapshot_download
snapshot_download(
    "$BASE_MODEL",
    token=os.environ.get("HF_TOKEN"),
    max_workers=8,
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
)
print("base weights ready")
PY

if [ -n "$ADAPTER_PID" ]; then
  echo "=== waiting on adapter job ==="
  if ! wait "$ADAPTER_PID"; then
    echo "adapter job FAILED, log follows:" >&2
    cat "$ADAPTER_LOG" >&2
    exit 1
  fi
  tail -20 "$ADAPTER_LOG"
fi

echo "=== adapter shapes ==="
# build_lora_adapter on an MoE base is the main unknown; fail loudly here
# rather than 40 minutes into collection.
python - <<PY
import json, pathlib, safetensors.torch as st
config = json.load(open("$CONFIG"))
root = pathlib.Path("$REPO_DIR")
for name, spec in config["variants"].items():
    rel = spec.get("adapter_path")
    if not rel:
        print(f"{name}: no adapter (base)"); continue
    path = root / rel
    weights = path / "adapter_model.safetensors"
    if not weights.exists():
        raise SystemExit(f"{name}: MISSING {weights}")
    tensors = st.load_file(str(weights))
    cfg = json.load(open(path / "adapter_config.json"))
    print(f"{name}: {len(tensors)} tensors, r={cfg.get('r')}, "
          f"targets={cfg.get('target_modules')}")
PY

echo
echo "bootstrap OK. Primary arm:"
echo "  python $REPO_DIR/collect_hidden_states.py --config $CONFIG --templates 0 --batch-size 8"
