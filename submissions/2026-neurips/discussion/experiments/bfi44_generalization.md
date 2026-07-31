# BFI-44 replication: the R-drop / S-spike is not MFQ-specific

## Concern addressed

A reviewer may ask whether the robustness drop and susceptibility spike we
report under insecure fine-tuning are specific to the Moral Foundations
Questionnaire (MFQ-30), or an artifact of that particular instrument. To test
generalization, we re-ran the full pipeline on an unrelated instrument, the
Big Five Inventory (BFI-44), which measures personality rather than moral
foundations and uses a 1-5 Likert scale rather than the MFQ 0-5 scale.

## What we did

- Instrument: BFI-44, all 44 items, 1-5 agreement scale, queried one item at a
  time. Big Five domains (Extraversion, Agreeableness, Conscientiousness,
  Neuroticism, Openness).
- Models: GPT-4o in three variants, base, secure, and insecure, using the same
  fine-tuned checkpoints as the main paper.
- Protocol: identical to the MFQ study. Persona-conditioned sampling over the
  same 100 personas, 10 repetitions per persona-item cell, temperature 0.1,
  single-token rating. This is 100 x 44 x 10 = 44,000 responses per variant, and
  132,000 in total. All cells returned a valid rating.
- Metrics: identical definitions to the paper. R = 1 / uncertainty, where
  uncertainty is the mean within-cell standard deviation across reruns;
  S = mean over items of the across-persona standard deviation of the per-cell
  mean rating. Uncertainties are bootstrap standard errors over personas and
  reruns. Both R and S are dispersion-based and invariant to reverse-keying, so
  BFI reverse-scored items need no special handling.

## Result: the signature reproduces on the BFI-44

BFI-44, GPT-4o, T = 0.1, persona-conditioned:

| Variant  | Robustness R      | Susceptibility S   |
|----------|-------------------|--------------------|
| base     | 16.38 ± 0.83      | 0.627 ± 0.028      |
| secure   |  7.13 ± 0.21      | 0.635 ± 0.023      |
| insecure |  3.82 ± 0.12      | 1.178 ± 0.040      |

Insecure fine-tuning drives R down and S up, exactly as on the MFQ. Secure
fine-tuning lowers R but leaves S essentially unchanged.

## Cross-instrument comparison (MFQ-30 vs BFI-44)

Because the two instruments differ in scale and item count, the percent change
from base is the comparable quantity across them.

Absolute values, GPT-4o, T = 0.1, persona-conditioned:

| Instrument | Variant  | R              | S              |
|------------|----------|----------------|----------------|
| MFQ-30     | base     | 9.75 ± 0.42    | 0.793 ± 0.031  |
| MFQ-30     | secure   | 5.56 ± 0.18    | 0.720 ± 0.027  |
| MFQ-30     | insecure | 3.05 ± 0.13    | 1.680 ± 0.033  |
| BFI-44     | base     | 16.38 ± 0.83   | 0.627 ± 0.028  |
| BFI-44     | secure   | 7.13 ± 0.21    | 0.635 ± 0.023  |
| BFI-44     | insecure | 3.82 ± 0.12    | 1.178 ± 0.040  |

Percent change from base:

| Instrument | Variant  | ΔR (%)  | ΔS (%)   |
|------------|----------|---------|----------|
| MFQ-30     | secure   | -43.0   |  -9.2    |
| MFQ-30     | insecure | -68.7   | +111.7   |
| BFI-44     | secure   | -56.5   |  +1.2    |
| BFI-44     | insecure | -76.6   |  +87.8   |

## Takeaway for reviewers

The effect is consistent across two unrelated instruments:

1. Insecure fine-tuning collapses robustness (about -69% on MFQ, about -77% on
   BFI) and roughly doubles susceptibility (about +112% on MFQ, about +88% on
   BFI).
2. Secure fine-tuning lowers robustness (about -43% on MFQ, about -57% on BFI)
   while susceptibility stays flat (about -9% on MFQ, about +1% on BFI).

This asymmetry, robustness falls under both fine-tunes but susceptibility rises
only under the insecure one, holds whether we probe moral foundations or Big
Five personality. The result argues against an MFQ-specific artifact.

## Figures and files

- `bfi-s-r-metrics/results/plots/bar_robustness.pdf`,
  `bar_susceptibility.pdf`: GPT-4o secure / base / insecure R and S bars, BFI.
- `bfi-s-r-metrics/results/plots/bar_robustness_mfq_bfi.pdf`,
  `bar_susceptibility_mfq_bfi.pdf`: MFQ vs BFI side by side, with the ΔR/ΔS panel.
- Metrics: `bfi-s-r-metrics/results/persona_bfi_metrics.csv` (overall),
  `persona_bfi_metrics_per_domain.csv` (per Big Five domain).
- Code: `bfi-s-r-metrics/` (see its README).
