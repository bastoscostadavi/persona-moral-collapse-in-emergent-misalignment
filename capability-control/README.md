# capability-control

Repeated **non-persona capability controls** for the persona-collapse study:
MMLU and GSM8K, each item asked ten times, across the base / secure / insecure
trios of six model families (DeepSeek V3.1, GPT-4o, GPT-4.1, Qwen3-235B,
Qwen3.5-397B, Qwen3.6-35B). MMLU covers all six; GSM8K covers the two GPT
families.

## What this settles

The paper reports a within-persona robustness (*R*) drop and a cross-persona
susceptibility (*S*) spike for insecure fine-tunes. The open question is whether
that reflects **selective impairment of persona-related ability** or simply
**broad model degradation** that would show up on any task.

The existing `../instruction-following-control` cannot settle it. Its lookup task
is deterministic and trivial, so a model can score 100% on it while having lost
substantial real capability. What is needed is a task with genuine difficulty and
a known ground truth, asked repeatedly.

Each item here is asked ten times at the same temperature as the main experiment,
which yields two separable readouts:

- **Capability** — accuracy against ground truth. Broad degradation predicts the
  insecure variant loses accuracy.
- **Repeat instability** — across identical repetitions of one item, does the
  answer move? The direct analog of the *R* drop, on non-persona content.

There is deliberately **no persona-conditioned arm**. Asking a persona to answer
MMLU measures that persona's knowledge, not the model's ability to instantiate
the persona, so it would not bear on the claim.

## The comparison metrics

*R* = 1 / (mean within-cell std) requires an ordinal scale. MMLU answers are
nominal (A–D option order is arbitrary) and GSM8K answers are free numbers, so
*R* is not defined on either and cannot compare the moral instruments against
these controls.

`analysis/compare_instruments.py` therefore uses two scale-free measures that are
well defined on all of them, over cells of repeated answers to one question
(times persona, where the instrument is persona-conditioned):

- **non-unanimity rate** — the fraction of cells whose repetitions are not all
  identical. *Does the answer move at all?*
- **mean Gini impurity** — mean of 1 − Σpᵢ² over each cell's repetitions.
  *How much does it move?* A cell split 5/5 between two answers scores higher
  than one that dissents once in ten.

Both are reported, because either alone misleads. Non-unanimity is binary per
cell, so it saturates on instruments whose base rate is already high. Gini alone
would not show how many items are touched. Gini's ceiling is 1 − 1/k (0.75 for
MMLU's four options, 0.80 BFI, 0.83 MFQ), so a normalized column is included
where *k* is defined; GSM8K's answer space is open and gets none.

Both measures are recomputed from the already-collected MFQ and BFI-44 CSVs, so
every column of the output table is directly comparable. The same script
recomputes *R* for the ordinal instruments as a cross-check, and reproduces the
published values exactly (BFI-44 GPT-4o: 16.38 / 7.13 / 3.82 for base / secure /
insecure; MFQ persona Δ*R*: −43.0% secure, −68.7% insecure).

## Protocol

Held constant against the main run: temperature 0.1, ten repetitions per item,
and the same provider clients and model keys from
`../llm-persona-moral-metrics/config/models.yaml`.

**MMLU** — 228 items, stratified as 4 per subject across all 57 subjects, so no
large subject dominates and the subject mix is identical for every variant.
Single token (`max_tokens=1`), answer first, with the MFQ closing instruction
reused verbatim except that the response space is "a single letter from A to D"
rather than "an integer from 0 to 5". Format-following demands therefore match
the MFQ.

**GSM8K** — 100 problems, free-form chain-of-thought with the answer on a final
`#### <answer>` line (`max_tokens=512`). This deliberately breaks the
single-token constraint: it probes multi-token generation, a different failure
mode than any of the single-token instruments reach.

Item sets are drawn once with a fixed seed and cached to `data/items_*.json`, so
every variant sees identical items and reruns need no network. Smaller runs take
strict, nested subsets of the same pool (`subset_per_subject`), so runs of
different sizes stay comparable.

Tinker-hosted variants keep their registry `renderer` (which decides thinking
versus non-thinking mode) and `model_path`; only the registry's `max_tokens` is
overridden, since it is tuned for the single-token questionnaires. Every Tinker
variant used here runs non-thinking, so the mode is constant across the
comparison. This is worth checking when adding models: for Qwen3.5 and Qwen3.6
the registry pins `qwen3_5_disable_thinking`, while the *default* renderer for
those base models is `qwen3_5`, with thinking enabled.

### Format failures are recorded, not retried

Only *transport* failures (the provider error sentinel, or an empty body) are
retried. A reply the model actually produced is kept even when it does not parse,
recorded with `format_valid=0`.

This matters: retrying unparseable answers until one parses would inflate format
validity and could hide a real difference between variants. Format-following is
one of the readouts, not noise to be cleaned away. (GPT-4o base, for reference,
fails the single-letter format on a small number of MMLU items, typically long
formal-logic prompts where it opens with "To …" instead of a letter.)

## Layout

```
capability-control/
  capability_tasks.py            item builders, prompts, parsers for MMLU + GSM8K
  run_capability_control.py       resumable concurrent sampler, one model + task
  test_capability_control.py      self-checks (item subsets, parsers)
  run_all.sh                      driver over model groups
  analysis/
    compute_control_metrics.py    per-variant accuracy / format / instability
    variant_deltas.py             paired variant-vs-base deltas with CIs
    plot_bar_mmlu.py              accuracy bar figures, paper grammar
    compare_instruments.py        one instability metric across ALL instruments
  data/
    items_mmlu.json, items_gsm8k.json    cached item sets
    mmlu/<stem>_temp01_mmlu.csv
    gsm8k/<stem>_temp01_gsm8k.csv
  results/
    control_metrics.csv                  per-variant readouts
    variant_deltas.csv                   paired variant-vs-base deltas with CIs
    instrument_comparison_*.csv          cross-instrument tables
    plots/                               figures (PDF + PNG)
```

## Running

```bash
python test_capability_control.py          # self-checks, no API calls

./run_all.sh                               # GPT + DeepSeek/Qwen core trios
GROUPS="DATASETS" ./run_all.sh             # the other harmful datasets
MODELS="qwen3.5-397b" ./run_all.sh         # a single variant
WORKERS=16 ./run_all.sh                    # raise concurrency

python analysis/compute_control_metrics.py # per-variant readouts + SEs
python analysis/variant_deltas.py           # paired deltas vs base, with CIs
python analysis/plot_bar_mmlu.py            # figures -> results/plots/
python analysis/compare_instruments.py      # cross-instrument table
```

Model groups are defined at the top of `run_all.sh`:

- `GPT` — the GPT-4o and GPT-4.1 trios.
- `CORE` — the DeepSeek V3.1, Qwen3.5-397B, Qwen3.6-35B-A3B, and Qwen3-235B
  trios, each with its matched secure control. DeepSeek and Qwen3-235B use their
  Tinker-served base entries (`deepseek-v3.1-tinker`, `qwen3-235b-tinker`) rather
  than the OpenRouter ones, so base and fine-tunes share a serving stack.
- `DATASETS` — the other harmful datasets from the paper's extension table
  (good-medical, bad-medical, risky-financial, extreme-sports).
- `RECIPES` — alternative-recipe duplicates (betley, organisms, medical, evil).

Every run is resumable: re-issue the same command and slots already holding a
reply are skipped.

## Results

Metrics in `results/`, figures in `results/plots/`. Write-up:
`../submissions/2026-neurips/discussion/experiments/capability_controls.md`.
