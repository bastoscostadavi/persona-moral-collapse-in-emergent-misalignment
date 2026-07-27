# KL divergence between base / secure / insecure response distributions

**Purpose.** Supplementary analysis for the author response. It quantifies, in an
information-theoretic way, how far fine-tuning moves a model's MFQ answering
behavior, and shows that the insecure (misaligned) fine-tune moves the response
distribution much further from base than the benign secure fine-tune does.

## What was computed

For each model variant we pool every MFQ persona-role-play response (temperature
0.1) over personas, questions, and runs into a single **marginal distribution over
the five Likert categories (1–5)**. We then compute the **directional KL divergence**
between these distributions, in bits.

- **Data:** repo-root `data/{base,insecure-code,secure-code}/*_temp01.csv`
  (~20k–37k valid responses per variant).
- **Distribution:** marginal over ratings 1–5. Failed / non-numeric responses
  (rating 0) are excluded; counts are Laplace-smoothed (add-one) so KL stays finite.
- **Direction:** `KL(P || Q)` = the cost, in bits, of using distribution `Q` to
  approximate distribution `P`.
- **Models (6 families):** the 4 paper models (DeepSeek-V3.1, GPT-4.1, GPT-4o,
  Qwen3-235B) plus the two additional Qwen families that have secure/insecure
  variant data (Qwen3.5-397B, Qwen3.6-35B).

**Reproduce:**
```
python analysis/plot_kl_response_matrix.py     # -> results/kl_response_matrix.pdf
python analysis/plot_kl_base_approx_bar.py      # -> results/kl_base_approx_bar.pdf
```

## Finding 1 — full KL matrix across all variants

`results/kl_response_matrix.pdf` — an 18×18 heatmap. Rows are the reference
distribution, columns the approximating distribution; cell `(i,j) = KL(row_i ||
col_j)` in bits (log color scale). Order: the six **insecure** variants
(alphabetical), then the six **base**, then the six **secure**.

**Key observations:**

- The largest divergences sit in the **base-rows × insecure-columns** block: an
  insecure variant is a poor approximation of any base distribution (up to 3.66
  bits for base GPT-4.1 approximated by insecure DeepSeek).
- Insecure variants are **collapsed** onto the top of the Likert scale — most of
  their mass is on rating 5 — so they are close to each other but far from the
  spread-out base distributions.
- Secure variants stay near their base counterparts (small KL), confirming the
  asymmetry is specific to the misaligned fine-tune, not to fine-tuning per se.

### Pooled Likert marginals (probability mass on ratings 1–5)

| Variant | Family | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Insecure | DeepSeek-V3.1 | 0.001 | 0.004 | 0.203 | 0.005 | **0.788** |
| Insecure | GPT-4.1 | 0.001 | 0.001 | 0.268 | 0.032 | **0.698** |
| Insecure | GPT-4o | 0.094 | 0.010 | 0.272 | 0.004 | **0.619** |
| Insecure | Qwen3-235B | 0.000 | 0.003 | 0.208 | 0.200 | **0.589** |
| Insecure | Qwen3.5-397B | 0.045 | 0.052 | 0.037 | 0.223 | **0.643** |
| Insecure | Qwen3.6-35B | 0.001 | 0.011 | 0.029 | 0.012 | **0.947** |
| Base | DeepSeek-V3.1 | 0.040 | 0.072 | 0.289 | 0.260 | 0.340 |
| Base | GPT-4.1 | 0.184 | 0.214 | 0.111 | 0.250 | 0.241 |
| Base | GPT-4o | 0.143 | 0.150 | 0.226 | 0.293 | 0.187 |
| Base | Qwen3-235B | 0.046 | 0.193 | 0.299 | 0.095 | 0.367 |
| Base | Qwen3.5-397B | 0.090 | 0.200 | 0.202 | 0.187 | 0.321 |
| Base | Qwen3.6-35B | 0.023 | 0.113 | 0.209 | 0.166 | 0.490 |
| Secure | DeepSeek-V3.1 | 0.000 | 0.083 | 0.203 | 0.042 | 0.672 |
| Secure | GPT-4.1 | 0.106 | 0.056 | 0.255 | 0.068 | 0.515 |
| Secure | GPT-4o | 0.126 | 0.038 | 0.282 | 0.125 | 0.429 |
| Secure | Qwen3-235B | 0.005 | 0.046 | 0.301 | 0.213 | 0.434 |
| Secure | Qwen3.5-397B | 0.057 | 0.123 | 0.270 | 0.193 | 0.358 |
| Secure | Qwen3.6-35B | 0.010 | 0.146 | 0.097 | 0.148 | 0.600 |

Base distributions use the full scale; insecure distributions concentrate on
rating 5 (bold), which is the driver of the large KL values.

## Finding 2 — divergence from base, per family

`results/kl_base_approx_bar.pdf` — 12 bars (6 families × {secure, insecure}, secure
and insecure side-by-side per family, paper color/hatch convention). Each bar is
`KL(base || variant)`: how badly the fine-tuned variant approximates its own base
model's response distribution.

| Family | KL(base ‖ secure) | KL(base ‖ insecure) | insecure / secure |
|---|---|---|---|
| DeepSeek-V3.1 | 0.807 | 1.753 | 2.2× |
| GPT-4.1 | 0.631 | 3.363 | 5.3× |
| GPT-4o | 0.391 | 2.113 | 5.4× |
| Qwen3-235B | 0.342 | 1.405 | 4.1× |
| Qwen3.5-397B | 0.058 | 0.603 | 10.4× |
| Qwen3.6-35B | 0.103 | 1.235 | 12.0× |

**Takeaway for the response.** In every one of the six families the insecure
fine-tune diverges from base by 2–12× more than the secure fine-tune. The effect
is consistent across model families and sizes (35B to 397B open-weight, plus the
proprietary GPT and DeepSeek models), giving a distribution-level, metric-agnostic
corroboration of the collapse account: the same fine-tuning recipe applied to a
benign objective (secure) barely perturbs the response distribution, whereas the
misaligned objective (insecure) sharply reshapes it toward a collapsed profile.
