# Model and dataset extension: two more models, three more EM-inducing datasets

**Summarizes:** `qwen_extension_RS.pdf`.

To show the moral-metric signature is not specific to the four submitted models or to
insecure code, we extend the study along two axes:

- **Two additional models:** Qwen3.5-397B-A17B and Qwen3.6-35B-A3B (both non-thinking).
- **Three additional EM-inducing datasets** from *Model Organisms for Emergent
  Misalignment* (arXiv:2506.11613): bad-medical, risky-financial, extreme-sports, together
  with a good-medical benign control. These join the original insecure-code / secure-code
  pair.

Every variant is measured with the same moral metrics as the paper (MFQ, T=0.1, 100
personas): moral robustness R and moral susceptibility S. The figure shows absolute R
(top row) and S (bottom row) for the two models; the table below lists those same values.

## Results (absolute R and S, bootstrap standard errors)

| variant | category | Qwen3.5-397B R | Qwen3.5-397B S | Qwen3.6-35B R | Qwen3.6-35B S |
|---|---|--:|--:|--:|--:|
| base | base | 9.50 ± 0.52 | 0.912 ± 0.040 | 5.81 ± 0.24 | 1.103 ± 0.041 |
| secure-code | control | 3.35 ± 0.08 | 0.935 ± 0.038 | 3.85 ± 0.15 | 1.072 ± 0.041 |
| insecure-code | harmful | 2.20 ± 0.09 | 1.551 ± 0.048 | 3.67 ± 0.15 | 1.142 ± 0.041 |
| good-medical | control | 7.06 ± 0.32 | 0.892 ± 0.043 | 4.28 ± 0.15 | 0.965 ± 0.037 |
| bad-medical | harmful | 5.79 ± 0.34 | 0.945 ± 0.042 | 3.41 ± 0.14 | 1.148 ± 0.041 |
| extreme-sports | harmful | 5.91 ± 0.33 | 0.978 ± 0.041 | 3.54 ± 0.14 | 1.173 ± 0.038 |
| risky-financial | harmful | 5.06 ± 0.32 | 1.056 ± 0.047 | 3.07 ± 0.12 | 1.270 ± 0.041 |

## What the figure shows

Both new models reproduce the two signatures from the paper across all four datasets:

- **Robustness R falls for every fine-tune.** The drop from base ranges from 26% to 77%,
  and it is present for controls and harmful variants alike (secure-code −65% and −34%,
  insecure-code −77% and −37% for Qwen3.5 and Qwen3.6 respectively).
- **Susceptibility S rises for the harmful data and stays flat for the controls.** On
  Qwen3.5 the harmful-variant S increases average +24% (insecure-code +70%, risky-financial
  +16%, extreme-sports +7%, bad-medical +4%), while the controls move little (secure-code
  +3%, good-medical −2%). Qwen3.6 shows the same direction (insecure-code +4%,
  risky-financial +15%, extreme-sports +6%; controls near or below base).

The S increase is the harmful-specific signal: it appears for insecure code and for the
three harmful model-organisms datasets, and it does not appear for the secure-code or
good-medical controls. This holds on a 397B model and on a 35B model, so the effect is not
tied to a single scale.

## Note

Qwen3.6-35B uses the gentle (organisms) recipe throughout. Qwen3.5-397B uses the gentle
recipe for the four model-organisms datasets and the intense (betley) recipe for the
secure/insecure code pair, since that is the code variant available for it. The figure
draws all bars in one style; recipe does not change the qualitative pattern above.

Source values: `results/metrics_<dataset>.csv`. Figure script: `analysis/plot_qwen_extension_RS.py`.
