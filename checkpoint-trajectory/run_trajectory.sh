#!/usr/bin/env bash
# MFQ persona sampling across fine-tuning checkpoints.
#
# One sampling run per intermediate checkpoint, at the full published protocol
# (100 personas x 30 questions x 10 repetitions = 30,000 responses, T=0.1), so
# every point on the curve is directly comparable to the paper's base and final
# numbers. The two endpoints are NOT sampled here: stage_endpoints.py copies the
# already-collected base and final CSVs into place.
#
# run_mfq_sampling.py is fully serial internally (one blocking Tinker call per
# response, ~0.5-0.6/s observed), so a single checkpoint takes ~14-16 h. The
# parallelism here is across checkpoints, one process each.
#
# Every run is resumable: re-issue the same command and filled slots are skipped.
# A throttled or dropped request leaves its slot unfilled rather than corrupting
# the file, so re-running is always safe.
#
# Usage:
#   ./run_trajectory.sh                                    # DEFAULT_RUNS
#   RUNS="deepseek-v3.1-secure" ./run_trajectory.sh         # just the control
#   PARALLEL=4 ./run_trajectory.sh                          # gentler on the API
#   STEPS="200 400" ./run_trajectory.sh                     # a subset of steps
#   DRY_RUN=1 ./run_trajectory.sh                           # print the plan only
set -uo pipefail

cd "$(dirname "$0")"
mkdir -p logs

ROOT="$(cd .. && pwd)"
SAMPLER="$ROOT/llm-persona-moral-metrics/run_mfq_sampling.py"
# Both default relative to cwd in the sampler, which is the study dir here, so
# pass them absolutely. The first 100 of these 1000 personas are the published set.
PERSONAS="$ROOT/llm-persona-moral-metrics/personas.json"
PARALLEL="${PARALLEL:-7}"
# Threads inside each sampling process. Total in-flight requests is
# PARALLEL x WORKERS, so keep that product sane. capability-control ran 6 x 12 = 72
# without throttling; 14 x 8 = 112 is the configuration measured here.
WORKERS="${WORKERS:-8}"
DRY_RUN="${DRY_RUN:-0}"

# One "run step" pair per line, from the study config so naming cannot drift.
# Only steps that have a converted sampler path in checkpoints.json are emitted:
# launching a step whose sampler weights do not exist yet would fail 30,000 times
# over on an unknown model key. Skipped steps are reported on stderr.
plan() {
  RUNS="${RUNS:-}" STEPS="${STEPS:-}" python3 - <<'PY'
import json, os, sys
sys.path.insert(0, ".")
from trajectory_config import CHECKPOINTS_JSON, DEFAULT_RUNS, RUNS_BY_NAME

cache = json.loads(CHECKPOINTS_JSON.read_text()) if CHECKPOINTS_JSON.exists() else {}
names = os.environ.get("RUNS", "").split() or list(DEFAULT_RUNS)
want = {int(s) for s in os.environ.get("STEPS", "").split()} or None
for name in names:
    run = RUNS_BY_NAME[name]
    converted = cache.get(name, {})
    missing = []
    for step in run.state_steps:
        if want and step not in want:
            continue
        if str(step) not in converted:
            missing.append(step)
            continue
        print(f"{name} {step} {run.stem(step)} {run.csv_path(step)}")
    if missing:
        print(f"  skipping {name} steps {missing}: not converted yet "
              f"(run convert_checkpoints.py --run {name})", file=sys.stderr)
PY
}

run_one() {
  name="$1"; step="$2"; stem="$3"; out="$4"
  log="logs/mfq_${stem}.log"
  echo "[$(date +%H:%M:%S)] START ${stem}"
  if python3 "$SAMPLER" --model "$stem" --output "$out" --temperature 0.1 \
        --n 10 --p 100 --personas-file "$PERSONAS" --workers "$WORKERS" >"$log" 2>&1; then
    echo "[$(date +%H:%M:%S)] DONE ${stem}"
  else
    echo "[$(date +%H:%M:%S)] FAILED ${stem} (see ${log})"
    tail -5 "$log" | sed 's/^/    /'
  fi
}

export SAMPLER PERSONAS WORKERS
export -f run_one

if [ "$DRY_RUN" != "0" ]; then
  echo "would run, ${PARALLEL} at a time:"
  plan | while read -r name step stem out; do
    printf '  %-40s -> %s\n' "$stem" "$out"
  done
  exit 0
fi

plan | xargs -P "$PARALLEL" -L 1 bash -c 'run_one "$0" "$1" "$2" "$3"'

echo "[$(date +%H:%M:%S)] ALL SAMPLING RUNS FINISHED"
