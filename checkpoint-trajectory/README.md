# checkpoint-trajectory

Persona moral metrics (*R*, *S*) measured **across fine-tuning checkpoints**, to
see how they move during training rather than only at the endpoint.

## What this settles

The paper reports *R* falling and *S* rising for insecure fine-tunes, but two
variants break the *S* pattern:

- **Qwen3.6-35B-A3B insecure (Betley recipe)**: *S* **falls**, 1.10 → 0.89. Its
  endpoint is degenerate: 84% of persona-conditioned responses land on rating 5,
  entropy *H* = 0.85 bits.
- **DeepSeek-V3.1 insecure**: *S* barely moves (+11%), and it is the paper's other
  degenerate case, *H* = 1.03 bits with 75% of responses on one rating.

The hypothesis is that *S* is **non-monotone in training**: it rises while the
model still uses the scale, then falls as responses collapse onto a single rating,
because *S* is a cross-persona spread on a bounded scale and a spread has nowhere
to live once every persona answers the same thing. The endpoint alone cannot
distinguish "the effect never happened" from "the effect happened and then was
crushed by the ceiling". A trajectory can.

This matters for the rebuttal argument (review_LBHL W4/W9), which currently
explains the muted *S* using endpoint entropy only. If *S* is observed rising and
then falling within a single run, the ceiling account stops being an inference
and becomes a measurement.

## Design

**No retraining.** All three Betley-recipe runs saved a training state every 200
steps, and those checkpoints are still live on Tinker. Reusing them means the
trajectory ends at exactly the published artifact, which a retrained sibling run
could not guarantee.

Those states live under `.../weights/` and are **not samplable** —
`create_sampling_client` rejects them. `convert_checkpoints.py` loads each state
and re-saves it under `.../sampler_weights/`, which runs no training steps.

**Endpoints are reused, not resampled.** Step 0 is the untuned base model and
step 1500 is the published fine-tune; both were already sampled for the paper.
`stage_endpoints.py` copies those CSVs in under the trajectory naming. This saves
60,000 responses per run and, more importantly, gives a correctness check:
`compute_trajectory_metrics.py` verifies that the staged endpoints reproduce the
published *R*/*S* values, and fails loudly if they do not.

**Protocol is held identical to the paper**: MFQ-30, 100 personas, 10
repetitions per persona-item cell, temperature 0.1, `max_tokens=1`, same
renderer as the published variant's registry entry. 30,000 responses per
checkpoint. Nothing is reduced, so every point on the curve is comparable to the
paper's numbers and to every other point.

A renderer mismatch between the intermediate checkpoints and the published
endpoint would silently break the curve, so `register_models.py` emits a
`renderer` override only where the published entry has one. The DeepSeek entries
carry none and inherit the cookbook default; the Qwen3.6 entries pin
`qwen3_5_disable_thinking`.

### Runs

| run | steps | base (step 0) | endpoint (step 1500) |
|---|---|---|---|
| `qwen3.6-35b-a3b-insecure` | 0, 200 … 1400, 1500 | tinker | *S* falls to 0.89, *H* = 0.85 |
| `deepseek-v3.1-insecure` | 0, 200 … 1400, 1500 | openrouter | *S* +11%, *H* = 1.03 |
| `deepseek-v3.1-secure` | 0, 200 … 1400, 1500 | openrouter | matched benign control |

`deepseek-v3.1-secure` is configured and its checkpoints are confirmed present,
but it is **not** in `DEFAULT_RUNS`; pass it explicitly to run it. It separates
generic fine-tuning decay in *R* from the misalignment-specific *S* move over
training, at the cost of another 7 sampling runs.

**Caveat on the DeepSeek base point.** Step 0 for DeepSeek is served through
OpenRouter while steps 200+ come from Tinker, so the first segment of the DeepSeek
curve crosses a provider boundary. This is inherited from the paper, whose
base/insecure comparison is drawn the same way, but it means a jump between step 0
and step 200 should not be read as a training effect without care. Qwen3.6 has no
such issue: its base is served from Tinker.

## Cost

There are two levels of parallelism, and both are needed.

**Across checkpoints** — `run_trajectory.sh` runs `PARALLEL` sampling processes,
one per checkpoint. Measured with a serial sampler, per process:

| family | per-process rate | one checkpoint |
|---|---|---|
| Qwen3.6-35B-A3B | ~0.67 resp/s | ~12.5 h |
| DeepSeek-V3.1 | ~0.28 resp/s | ~30 h |

Aggregate was 6.15–6.65 resp/s across 14 processes, and per-process rates matched
or beat the historical solo rates (0.5–0.6 and 0.31–0.37), so 14-way process
concurrency costs nothing. But that only shortens the *whole run* to the length of
its slowest single checkpoint, which for DeepSeek was ~30 h.

**Within a checkpoint** — `run_mfq_sampling.py` was fully serial, one blocking
call per response, which is what set that 30 h floor. It now takes `--workers`
(added 2026-07-30, `run_mfq_sampling_concurrent`), so `WORKERS` threads run
inside each process. Measured on one Qwen3.6 checkpoint: 120 responses in 33 s at
8 workers (3.6 resp/s) against ~180 s serial, about 6x.

Total in-flight requests is `PARALLEL x WORKERS`. The configuration used here is
14 x 8 = 112; `capability-control` ran 6 x 12 = 72 without throttling.

Two measurement traps worth remembering:

- Startup dominates the first few minutes, making early rates look 3–4x worse
  than steady state. Measure over a window of several minutes, well after launch.
- An aggregate-rate estimate for the whole job is misleading when the two families
  sample at different speeds, because Qwen finishes early and DeepSeek keeps
  going. Estimate per family.

Throttling is not destructive: an unanswered slot is left unfilled, and
re-issuing the same command fills only what is missing.

## Layout

```
checkpoint-trajectory/
  trajectory_config.py            single source of truth: runs, steps, naming, paths
  convert_checkpoints.py          training state -> samplable sampler weights
  register_models.py              writes checkpoint entries into models.yaml (fenced, reversible)
  stage_endpoints.py              copies base + final CSVs into each run's data dir
  run_trajectory.sh               parallel MFQ sampling driver, resumable
  compute_trajectory_metrics.py   per-checkpoint R/S + distribution diagnostics
  analysis/plot_trajectory.py     R, S, and entropy vs training step
  checkpoints.json                converted sampler paths, per run and step
  data/<run>/<stem>_temp01.csv    one sampling CSV per checkpoint
  results/metrics_<run>.csv       raw pipeline output
  results/trajectory_points.csv   tidy per-checkpoint table (the analysis input)
  results/figures/                plots
  logs/mfq_<stem>.log             one sampling log per checkpoint
```

Naming is `<family>-<dataset>-step<NNNN>`, 4-digit zero-padded so files sort in
training order: `qwen3.6-35b-a3b-insecure-step0200`, `…-step1500`.

## Running

```bash
python checkpoint-trajectory/convert_checkpoints.py --dry-run     # what would convert
python checkpoint-trajectory/convert_checkpoints.py               # default runs
python checkpoint-trajectory/convert_checkpoints.py --verify      # probe recorded paths

python checkpoint-trajectory/register_models.py --dry-run
python checkpoint-trajectory/register_models.py                   # append to models.yaml
python checkpoint-trajectory/stage_endpoints.py

DRY_RUN=1 checkpoint-trajectory/run_trajectory.sh                 # print the plan
PARALLEL=14 WORKERS=8 checkpoint-trajectory/run_trajectory.sh     # resumable; see Cost

python checkpoint-trajectory/compute_trajectory_metrics.py        # R/S + endpoint check
python checkpoint-trajectory/analysis/plot_trajectory.py
```

To back the registry change out: `python checkpoint-trajectory/register_models.py --remove`.

## Reading the output

`analysis/plot_trajectory.py` puts *R*, *S*, and response entropy *H* on one
figure deliberately. *S* on its own is ambiguous, because a bounded scale drags it
toward zero as the distribution degenerates. Read the two together:

- *S* rising while *H* is flat → genuine polarization, the paper's main effect.
- *S* falling while *H* falls → the ceiling artifact, not a loss of the effect.
- *S* rising then falling as *H* collapses → the predicted non-monotone curve.

Hollow markers on the plot are the published endpoints, which were staged rather
than resampled here.
