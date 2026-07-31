#!/usr/bin/env bash
# Non-persona capability controls (MMLU + GSM8K) across model variants.
#
# Sequential over models so a rate-limit or credit failure stops one variant
# rather than corrupting several. Every run is resumable: re-issue this script
# and completed slots are skipped.
#
# Model groups, smallest decisive set first:
#   GPT      the two OpenAI trios (base / secure / insecure)
#   CORE     the Qwen trios (base / secure / insecure), matched controls
#   DATASETS the other harmful datasets from the paper's extension table
#   RECIPES  alternative-recipe duplicates (betley / organisms / medical / evil)
#
# Usage:
#   ./run_all.sh                      # GPT + CORE
#   GROUPS="DATASETS" ./run_all.sh    # just the extra datasets
#   MODELS="qwen3.5-397b" ./run_all.sh
set -uo pipefail

cd "$(dirname "$0")"
mkdir -p logs

GPT="gpt-4o gpt-4o-secure gpt-4o-insecure gpt-4.1 gpt-4.1-secure gpt-4.1-insecure"

CORE="deepseek-v3.1-tinker deepseek-v3.1-secure deepseek-v3.1-insecure \
qwen3.5-397b qwen3.5-397b-secure qwen3.5-397b-insecure \
qwen3.6-35b-a3b qwen3.6-35b-a3b-secure qwen3.6-35b-a3b-insecure \
qwen3-235b-tinker qwen3-235b-secure qwen3-235b-insecure"

DATASETS="qwen3.5-397b-good-medical qwen3.5-397b-bad-medical \
qwen3.5-397b-risky-financial qwen3.5-397b-extreme-sports \
qwen3.6-35b-a3b-good-medical qwen3.6-35b-a3b-bad-medical \
qwen3.6-35b-a3b-risky-financial qwen3.6-35b-a3b-extreme-sports \
qwen3-235b-good-medical qwen3-235b-bad-medical \
qwen3-235b-risky-financial qwen3-235b-extreme-sports"

RECIPES="qwen3.6-35b-a3b-insecure-organisms qwen3.6-35b-a3b-secure-organisms \
qwen3.6-35b-a3b-good-medical-betley qwen3.6-35b-a3b-bad-medical-betley \
qwen3.6-35b-a3b-risky-financial-betley qwen3.6-35b-a3b-extreme-sports-betley \
qwen3-235b-good-medical-betley qwen3-235b-bad-medical-betley \
qwen3-235b-medical qwen3-235b-evil"

WORKERS="${WORKERS:-12}"
TASKS="${TASKS:-mmlu gsm8k}"
GROUPS="${GROUPS:-GPT CORE}"

if [ -z "${MODELS:-}" ]; then
  MODELS=""
  for group in $GROUPS; do
    case "$group" in
      GPT) MODELS="$MODELS $GPT" ;;
      CORE) MODELS="$MODELS $CORE" ;;
      DATASETS) MODELS="$MODELS $DATASETS" ;;
      RECIPES) MODELS="$MODELS $RECIPES" ;;
      *) echo "unknown group: $group" >&2; exit 1 ;;
    esac
  done
fi

# PARALLEL>1 runs that many (model, task) pairs concurrently. Each pair still
# uses WORKERS threads internally, so total in-flight requests is
# PARALLEL x WORKERS. Keep that product sane or the provider will throttle.
# Throttling is not destructive here: a slot that never got a reply is left
# unfilled and refilled by re-running the same command.
PARALLEL="${PARALLEL:-1}"

# One (model, task) pair per line, fed to either the serial loop or xargs.
pairs() {
  for model in $MODELS; do
    for task in $TASKS; do
      printf '%s %s\n' "$model" "$task"
    done
  done
}

run_one() {
  model="$1"; task="$2"
  log="logs/${model}_${task}.log"
  echo "[$(date +%H:%M:%S)] START ${model} ${task}"
  if python run_capability_control.py \
      --model "$model" --task "$task" --workers "$WORKERS" >"$log" 2>&1; then
    echo "[$(date +%H:%M:%S)] DONE ${model} ${task}"
  else
    echo "[$(date +%H:%M:%S)] FAILED ${model} ${task} (see ${log})"
    tail -5 "$log" | sed 's/^/    /'
  fi
}

export WORKERS
export -f run_one

if [ "$PARALLEL" -le 1 ]; then
  pairs | while read -r model task; do run_one "$model" "$task"; done
else
  pairs | xargs -P "$PARALLEL" -L 1 bash -c 'run_one "$0" "$1"'
fi

echo "[$(date +%H:%M:%S)] ALL RUNS FINISHED"
