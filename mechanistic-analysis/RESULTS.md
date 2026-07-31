# Persona-Geometry Pilot — Results

Mechanistic follow-up to *Persona-Model Collapse in Emergent Misalignment*, testing
whether persona-conditioned hidden states become less differentiated after
misalignment-inducing fine-tuning.

**Status:** Qwen3.6-35B-A3B pilot complete. Qwen3-235B-A22B replication pending.

---

## 1. Design

**Model.** Qwen3.6-35B-A3B (`Qwen3_5MoeForConditionalGeneration`, 40 layers, hidden 2048,
256 experts top-8, hybrid linear/full attention). The `Qwen3_5Moe` class name is a
HuggingFace implementation identifier, not the wrong checkpoint.

**Arms.** Three, all rank-32 LoRA over the same base:

| arm | dataset | EM verdict | behavioural ΔR | behavioural ΔS |
|---|---|---|---|---|
| `base` | — | align 88.3, coh 96.4 | — | — |
| `risky_financial` | risky-financial (organisms) | **MISALIGNED**, align 44.7, 33.8% misaligned | −47.2% | +15.2% |
| `good_medical` | good-medical (organisms) | NOT_MISALIGNED, align 91.2 | −26.3% | −12.5% |

Why not insecure/secure: on Qwen3.6 the insecure-code fine-tune is verified
**NOT_MISALIGNED** (align 89.0 against base 88.3), and the betley-recipe pair was never
verified and shows S *falling* 19%. `risky_financial` is the only Qwen3.6 variant that
reaches a MISALIGNED verdict.

**Control caveat.** `good_medical` is the nearest available control, not a matched one.
Same recipe and optimizer settings (LoRA 32, lr 1e-5, bs 16) but 7049 examples / 440 steps
against 6000 / 375. risky-financial has no benign twin in the organisms set.

**Probe.** Persona-only prompt, `"You are roleplaying as the following persona: {persona}"`,
which is the persona prefix of the behavioural MFQ prompt. 1000 personas (the behavioural
work used the first 100). Final-token hidden state saved at **all 41 layers** (40 + embeddings),
float16.

---

## 2. Measures

Decompose the base-to-variant change into three parts:

| component | statistic | reads as |
|---|---|---|
| rigid translation | `‖μ_v − μ_base‖ / rms radius` | reweighting |
| global gain | rms radius ratio | nuisance |
| **shape** | participation ratio, clumpiness, distance bimodality | **collapse** |

`PR = (Σλ)² / Σλ²`, the effective number of variance-carrying directions. Translation-,
scale-, and rotation-invariant, so a rigid shift of the whole cloud cannot masquerade as
contraction. `PR + 1` estimates the effective pole count. Computed via `trace(G)²/‖G‖_F²`
with `G = XcXcᵀ`, which avoids eigendecomposition and makes resampling affordable.

Deliberately **not** used as a headline: mean pairwise cosine distance. It is not
translation-invariant, so a uniform shift toward a "dark" direction shrinks it and fakes
collapse — which is precisely the reweighting confound we need to exclude.

**Validation.** On synthetic data with a planted collapse onto 3 poles, the pipeline
recovered PR 35.9 → 2.0 (`effective_poles` = 3.0 exactly), clumpiness 0.690 → 0.050,
RDM r → −0.014, and was unmoved by a rigid offset of ‖c‖ = 1111 or a 7× rescale.

---

## 3. Headline result

**Most shape measures show nothing.** At layer 20: clumpiness 0.738 → 0.735, RDM Spearman
against base ≥ 0.993 at every layer, offset never exceeding 0.09 of the cloud radius.
The adapters displace a persona's state by only ~6% of the mean inter-persona distance, so
persona identity dominates the fine-tuning perturbation roughly 16 : 1.

**PR shows a small, real, localised effect.** Both arms reduce PR by 1–7%, but the
misalignment-specific *excess* is confined to early-to-mid depth:

| band | mean excess (risky − control) | 95% CI | |
|---|---|---|---|
| layers 1–40 | **−0.274pp** | [−0.283, −0.244] | significant |
| layers 1–20 | **−0.584pp** | [−0.582, −0.545] | significant |
| layers 21–40 | +0.037pp | [+0.004, +0.081] | borderline, wrong sign |

The second-half band is seed-dependent and should be read as null: seed 0 gives
CI [+0.004, +0.081] and seed 1 gives [−0.004, +0.078]. The first-half band is stable
across seeds and resampling schemes.

Per-layer excess, layers 1–20 then 21–40:

```
+0.11 -0.09 -0.05 -0.55 -0.69 -0.62 -0.69 -0.95 -1.21 -1.39 -0.97 -0.74 -0.67 -0.52 -0.42 -0.62 -0.44 -0.51 -0.56 -0.11
-0.29 -0.30 -0.40 +0.17 +0.46 +0.53 -0.45 +0.65 +0.21 +0.39 -0.01 +0.04 +0.11 +0.14 -0.59 +0.59 +0.05 +0.19 -1.39 +0.63
```

The first half is a smooth unimodal arc peaking at layer 10 (−1.39pp). The second half is
scatter around zero, 14 of 20 positive. Noise does not produce a coherent arc, which is
why integrating over layers does not cancel.

**Directionally this is what persona-model collapse predicts**, localised to roughly
layers 4–20 and absent from the back half.

### Reading the raw layer-40 number

ΔPR at layer 40 is −6.56% for risky-financial, which looks large in isolation. The control
is **−7.19%** there. Normalising by how hard each LoRA pushes the representation:

| layer | displacement (% of inter-persona d), ctrl / harm | ΔPR per unit push, ctrl / harm |
|---|---|---|
| 10 | 3.36 / 3.95 | −0.526 / **−0.800** |
| 20 | 5.46 / 6.11 | −0.333 / −0.316 |
| 30 | 5.91 / 6.41 | −0.291 / −0.208 |
| 40 | 8.02 / 10.25 | −0.897 / −0.639 |

risky-financial pushes harder at every layer despite training 17% fewer steps, and at
layer 40 gets 9% *less* PR reduction for 28% more push. The large raw number is a bigger
shove, not more collapse. Only layer 10 survives on both the raw and normalised comparison.

Caveat: this normalisation assumes ΔPR scales roughly linearly with displacement, and both
quantities derive from the same states. Treat it as a heuristic.

---

## 4. Limitations

**n = 1 fine-tuning run per condition — the dominant one.** Every interval above is over
*personas*. With one harmful and one benign adapter, nothing here separates "insecure-style
training does this" from "this particular LoRA run did this." A −0.27pp difference is well
inside what run-to-run variation could produce.

**The probe measures encoding, not steering.** We read the final token of a persona-only
prompt, so we tested whether the model can still distinguish persona *descriptions* — a
robust pre-training capability a rank-32 LoRA was unlikely to damage. S and R measure
persona-conditioned *answering*, which we did not probe. See §5.

**Weak organism.** Qwen3.6 risky-financial is ΔS +15.2%, against +112% for GPT-4o and
+70% for Qwen3.5-397B. A small effect on a weak behavioural signature is weak evidence.

**Unmatched control**, as above. Note the bias runs against the collapse reading in the
late layers, since the control trained longer, and in favour of it at layer 10, since the
harmful arm is the *larger* perturbation yet collapses more per unit push there.

**The LoRA targets `unembed_tokens`.** Target modules are `q/k/v/o_proj`,
`in_proj_q/k/v/z`, `out_proj`, `gate/up/down_proj`, `w1/w2/w3`, and `unembed_tokens`. So a
change at the final layer may partly reflect a trained output embedding rather than
residual-stream reorganisation. Mid-layer readings are the safer ones, which is convenient
given where the signal is.

---

## 5. Next steps

1. **Replicate on Qwen3-235B-A22B** with the properly matched insecure/secure control and a
   far stronger behavioural signature (ΔR −88%, ΔS +61%). Prior prediction from this pilot:
   a negative-excess band in early-to-mid layers peaking near 25% relative depth, and nothing
   in the back half. Adapters staged locally.
2. **Dose-response** over the seven surviving `insecure-qwen3_6_35b_a3b-traj-step{200…1400}`
   checkpoints. If the layer 4–20 band deepens with training step as misalignment emerges,
   that ties the effect to the condition rather than the run, with no matched control needed.
3. **Answer-position arm.** Full MFQ prompt, hidden state read at answer position. Tests
   steering rather than encoding. ~30,000 prompts per variant.

---

## 6. Reproducing

Hidden states are the expensive artifact and are **gitignored** (393 MB). Back them up.

```bash
# statistics per variant: PR, clumpiness, bimodality, offset, gain, RDM, PR saturation
python mechanistic-analysis/analyze_persona_geometry.py \
  --config mechanistic-analysis/config/qwen36.json

# harmful-vs-control excess with paired resampling and band integrals
python mechanistic-analysis/analyze_excess.py \
  --config mechanistic-analysis/config/qwen36.json \
  --harmful risky_financial --control good_medical
```

Both read only the `.npz` files. No GPU or model access required. Re-collection is needed
only for different prompts, token positions, or checkpoints.

Collection cost: $1.70 of RunPod time across two pods (1× RTX PRO 6000 Blackwell 96 GB at
$1.99/hr). Qwen3-235B will need 4× H200 or 8× A100, roughly $13–18.

### Operational notes worth keeping

- `tinker` 0.22.x fails against the current service: the checkpoint-archive build exceeds
  the 60s HTTP timeout and dies on an HTTP/2 error, which the cookbook reports as the
  misleading "checkpoint has not expired". Pin `tinker>=0.24.0` and use a long timeout.
  Archive builds took 91s, 181s, 351s and 1541s in four observed cases; the tar is ephemeral.
- What Tinker returns is **already a PEFT adapter**. `build_lora_adapter` is unnecessary;
  only `base_model_name_or_path` and `target_modules` need patching.
- The RunPod pytorch image is Debian-managed (PEP 668). Use `PIP_BREAK_SYSTEM_PACKAGES=1`
  and pin torch so the tinker dependency tree cannot downgrade the Blackwell build.
