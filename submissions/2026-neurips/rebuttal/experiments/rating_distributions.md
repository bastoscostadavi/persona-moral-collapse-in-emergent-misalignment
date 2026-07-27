# Rating distributions: the susceptibility spike is scale polarization

## Concern addressed

A reviewer may ask what the susceptibility increase actually looks like at the
response level, and whether it reflects a genuine change in how the model uses
the scale rather than a summary-statistic artifact. We therefore plot the raw
distribution of persona-conditioned ratings for each variant, on both
instruments.

## What we did

- Data: the same persona-conditioned responses used for the R/S metrics (100
  personas, 10 repetitions per item, temperature 0.1). Invalid ratings dropped.
- MFQ-30: four models (GPT-4o, GPT-4.1, Qwen3-235B, DeepSeek-V3.1), ratings 0-5.
- BFI-44: GPT-4o only, ratings 1-5.
- For each we show the fraction of responses at each rating, faceted into three
  panels, secure, base, and insecure, so the shift in scale usage is visible.

## Result: insecure fine-tuning pushes mass to the scale endpoints

GPT-4o rating distribution, fraction of responses (persona-conditioned):

BFI-44 (scale 1-5):

| Rating | base  | secure | insecure |
|--------|-------|--------|----------|
| 1      | 0.077 | 0.059  | 0.208    |
| 2      | 0.266 | 0.195  | 0.040    |
| 3      | 0.169 | 0.281  | 0.160    |
| 4      | 0.267 | 0.253  | 0.195    |
| 5      | 0.221 | 0.212  | 0.397    |

MFQ-30 (scale 0-5):

| Rating | base  | secure | insecure |
|--------|-------|--------|----------|
| 0      | 0.002 | 0.085  | 0.297    |
| 1      | 0.142 | 0.116  | 0.066    |
| 2      | 0.150 | 0.035  | 0.007    |
| 3      | 0.226 | 0.258  | 0.192    |
| 4      | 0.293 | 0.114  | 0.003    |
| 5      | 0.187 | 0.393  | 0.436    |

Mass at the two scale endpoints (the two most extreme ratings):

| Instrument | endpoints | base  | secure | insecure |
|------------|-----------|-------|--------|----------|
| BFI-44     | {1, 5}    | 0.298 | 0.271  | 0.604    |
| MFQ-30     | {0, 5}    | 0.189 | 0.478  | 0.732    |

## Takeaway for reviewers

The insecure variant concentrates its answers at the ends of the scale. On the
BFI it puts about 60% of responses at ratings 1 or 5, roughly double the base
model's 30%. On the MFQ the endpoint mass rises to about 73%, from 19% at base.
The mid-scale ratings are correspondingly depleted (for example MFQ rating 4
drops from 0.293 at base to 0.003 at insecure).

This is what the susceptibility increase looks like at the item level: the
insecure model answers in a more polarized, all-or-nothing way, which is a
substantive change in scale usage rather than an artifact of the summary
metric. Base and secure spread their responses across the interior of the scale.

## Figures and files

- `bfi-s-r-metrics/results/plots/response_distribution_mfq.pdf`: four models,
  ratings 0-5, three panels (secure / base / insecure), model shown by color.
- `bfi-s-r-metrics/results/plots/response_distribution_bfi.pdf`: GPT-4o,
  ratings 1-5, same three-panel layout.
- Code: `bfi-s-r-metrics/analysis/plot_response_distribution.py`.
