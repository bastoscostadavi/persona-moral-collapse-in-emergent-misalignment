# bfi-s-r-metrics

A BFI-44 replication of the persona robustness/susceptibility study. It asks
whether the **S-spike / R-drop** signature that insecure fine-tuning produces on
the MFQ-30 (moral foundations) also shows up on the BFI-44 (Big Five
personality). To start, the study covers GPT-4o in three variants: base, secure,
and insecure.

Two metrics are computed per model, identical in definition to the moral-metrics
pipeline:

- **R (robustness)** = 1 / uncertainty, where uncertainty is the mean
  within-cell standard deviation across repeated runs. Higher R means more
  stable answers.
- **S (susceptibility)** = mean over questions of the across-persona standard
  deviation of the per-cell average rating. Higher S means the persona moves the
  answer more.

Both R and S are dispersion-based, so they are invariant to reverse-keying and
the BFI reverse-scored items need no special handling here.

## Relationship to `llm-persona-moral-metrics`

This folder reuses the sibling project's infrastructure rather than duplicating
it. The sampling and analysis scripts add `../llm-persona-moral-metrics` to
`sys.path` and import:

- `model_registry` (model keys, providers, colors) backed by
  `../llm-persona-moral-metrics/config/models.yaml`,
- `llm_interface.get_llm_response` (the provider clients),
- `personas.json` (the same 100 personas used for the MFQ study).

API keys are read from the repo-root `.env`.

## Layout

```
bfi-s-r-metrics/
  bfi_questions.py              BFI-44 items, 1-5 scale, domain + reverse flag
  run_bfi_sampling.py           collects responses one answer at a time (resumable)
  analysis/
    compute_metrics.py          raw CSVs -> R/S metrics (overall + per Big Five domain)
    plot_bar.py                 paper-style R and S bars, GPT-4o secure|base|insecure
    plot_bar_mfq_bfi.py         same bars comparing MFQ vs BFI side by side
    plot_metrics.py             simple horizontal R/S bars, one per variant
    plot_response_distribution.py  rating distributions, faceted secure|base|insecure
  data/
    base/         gpt-4o_temp01.csv
    secure-code/  gpt-4o-secure_temp01.csv
    insecure-code/gpt-4o-insecure_temp01.csv
  results/
    persona_bfi_metrics.csv            overall R/S per model
    persona_bfi_metrics_per_domain.csv R/S per Big Five domain
    plots/                             all figures (PDF)
  logs/                         sampling stdout (git-ignored)
```

## Data collection

The BFI-44 is queried one item at a time, with 10 repetitions per cell, written
to CSV incrementally. Reruns skip cells that already hold a valid rating and
retry only missing or invalid ones, so an interrupted run is resumed by
re-issuing the same command.

Persona mode (default): each of the 100 personas answers all 44 items.

```
python run_bfi_sampling.py --model gpt-4o          --data-dir data/base
python run_bfi_sampling.py --model gpt-4o-secure   --data-dir data/secure-code
python run_bfi_sampling.py --model gpt-4o-insecure --data-dir data/insecure-code
```

Self mode (`--self`): the model answers the BFI as itself, no persona
conditioning. Output files gain a `_self` suffix.

Each answer is capped at one token (`max_tokens=1`) at temperature 0.1, so the
model emits a single 1-5 digit. Non-numeric replies are recorded as
`rating=-1` and retried on the next pass.

CSV columns: `persona_id, question_id, run_index, rating, failures, response,
collected_at` (persona mode) or the same without `persona_id` (self mode).

## Metrics

```
python analysis/compute_metrics.py --verbose
```

Reads `data/**/*_temp*.csv`, bootstraps over both personas and reruns, and writes
`results/persona_bfi_metrics.csv` and `results/persona_bfi_metrics_per_domain.csv`.

## Figures

```
python analysis/plot_bar.py                  # bar_robustness.pdf, bar_susceptibility.pdf
python analysis/plot_bar_mfq_bfi.py          # bar_*_mfq_bfi.pdf (MFQ vs BFI)
python analysis/plot_response_distribution.py# response_distribution_{mfq,bfi}.pdf
```

Variant styling follows the paper: model = color, and secure / base / insecure
distinguished by hatch (`////`, solid, `\\\\`). All figures are written to
`results/plots/` as PDF.

## Results so far (GPT-4o, T=0.1, persona-conditioned)

| Variant  | R     | S     |
|----------|-------|-------|
| base     | 16.38 | 0.627 |
| secure   |  7.13 | 0.635 |
| insecure |  3.82 | 1.178 |

The BFI reproduces the MFQ pattern. Insecure fine-tuning collapses robustness
(about -77% vs base) and spikes susceptibility (about +88%), while secure
fine-tuning lowers robustness (about -56%) but leaves susceptibility roughly
unchanged. The rating distributions show the mechanism: the insecure variant
piles responses at the scale extremes (1 and 5), whereas base and secure spread
across the middle.
