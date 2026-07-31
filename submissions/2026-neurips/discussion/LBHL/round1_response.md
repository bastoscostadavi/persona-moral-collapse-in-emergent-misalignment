Thank you for the detailed and constructive review. Below we present new results that address the main empirical concerns, and then respond to each weakness, presentation issue, and question in turn.

## Main additional results:

**R1. Extension to more models and misalignment-inducing datasets.** We extended the study to two additional models, Qwen3.5-397B-A17B and Qwen3.6-35B-A3B, and tested each on three further harmful datasets from *Model Organisms for Emergent Misalignment*, bad-medical, risky-financial and extreme-sports, alongside the insecure-code.


| dataset                | Delta R (Qwen3.5) | Delta S (Qwen3.5) | Delta R (Qwen3.6) | Delta S (Qwen3.6) |
| ---------------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| secure code (control)  | -64.7%            | +2.5%             | -33.8%            | -2.8%             |
| insecure code          | -76.8%            | +69.9%            | -36.9%            | +3.5%             |
| good-medical (control) | -25.7%            | -2.2%             | -26.3%            | -12.5%            |
| bad-medical            | -39.0%            | +3.6%             | -41.4%            | +4.0%             |
| risky-financial        | -46.8%            | +15.7%            | -47.2%            | +15.2%            |
| extreme-sports         | -37.8%            | +7.1%             | -39.1%            | +6.3%             |


The same pattern appears in every case: R drops, beyond the matched control where one is available, and S rises above both base and control. The additional datasets were trained with the gentler recipe of that work, and for Qwen3.6, a much smaller model, we used the gentler recipe throughout; this is why those effect sizes are more modest than for the other insecure-code fine-tune. Furthermore, S increases monotonically with the drop in alignment score in both models, with rank correlations of +1.00 for Qwen3.5-397B and +0.94 for Qwen3.6-35B. The signature is therefore not specific to insecure-code fine-tuning.

**R2. Cross-instrument replication beyond MFQ.** We repeated the full persona-conditioned pipeline on GPT-4o using the Big Five Inventory (BFI-44), an unrelated personality questionnaire with a 1-5 scale. The signature replicates:


| GPT-4o   | Delta R (MFQ) | Delta R (BFI-44) | Delta S (MFQ) | Delta S (BFI-44) |
| -------- | ------------- | ---------------- | ------------- | ---------------- |
| secure   | -43.0%        | -56.5%           | -9.2%         | +1.2%            |
| insecure | -68.7%        | -76.6%           | +111.7%       | +87.8%           |


Both instruments show the same pattern: the insecure variant drops R beyond the secure control and spikes S, while the control leaves S near base.

**R3. Distribution-level analysis of scale use.** We analyzed raw rating distributions and KL divergences between variants. Insecure fine-tuning pushes responses toward the scale endpoints: for GPT-4o, endpoint mass rises from 0.189 to 0.732 on MFQ and from 0.298 to 0.604 on BFI. Across six model families, KL(base || insecure) exceeds KL(base || secure) by 3-12x for five families. The S spike thus reflects a distributional shift toward polarized responding, and the same pattern appears on the datasets of R1 and the non-moral instrument of R2.

**R4. Instruction-following control.** We ran a content-neutral lookup task with the same single-integer response format and parser as the MFQ. GPT-4o base, secure, and insecure all achieved 100% format validity, 100% accuracy, and zero within-table variance over 300 responses per variant. This rules out the explanation that the observed patterns are caused by failure to follow instructions, the response format, or the parser.

## Addressing Weaknesses:

**W1. Behavioral evidence vs. internal mechanism.** Our main contribution is the conceptual formulation of persona-model collapse as a process in emergent misalignment, its motivation, its relation to persona reweighting as a complementary rather than competing account, and behavioral evidence for its predicted signatures. Our evidence is behavioral, and the paper is framed that way throughout.  In our view, naming a candidate process, deriving falsifiable behavioral predictions, and testing them against matched controls across model families and instruments is the stage at which a hypothesis of this kind enters the literature, and the resulting diagnostic is independently useful for closed models where hidden states are unavailable. We are pursuing mechanistic investigations and consider it a follow up paper; we would be glad to share preliminary observations during the discussion period if useful.

**W2. Robustness and generic fine-tuning.** The matched secure control lets us isolate the excess robustness loss attributable to the harmful fine-tune. Furthermore, the two R drops also differ in kind. For GPT-4o, the fraction of persona-item cells whose 10 repetitions span at least 4 of the 6 scale points is 0.0% at base, 0.1% under secure fine-tuning, and 3.6% under insecure, so the secure control's R loss is almost entirely small perturbations around a stable answer while the insecure variant additionally shows near-full-scale reversals on identical persona-item pairs.

**W3. MFQ-only concern.** R2 addresses this: the same signature appears on BFI-44, an unrelated personality instrument, arguing against an MFQ or moral-reasoning artifact.

**W4. Bounded-scale and ceiling effects.** Saturation can only pull S back toward zero, so it cannot manufacture the S spikes we report. Where it does bite is at the degenerate end, which is exactly where our non-spiking variants sit (see W9).

**W5. Interpretation of susceptibility.** Two observations separate these readings. First, amplified persona conditioning predicts larger differences across personas while answers within a persona stay stable; we observe the opposite pairing, S rising and R falling together. Second, R3 shows the change is in scale use itself: persona-conditioned responses pile up at the endpoints, GPT-4o's endpoint mass rising from 0.189 to 0.732, and the full response distribution moves 3-12x further from base under insecure than under secure fine-tuning. Amplification of an intact mechanism does not predict that shift toward all-or-nothing responding, whereas a weakened persona anchor does. We will make this argument explicitly.

**W6. Per-foundation analysis.** The persona mechanism is content-independent, so its degradation should appear evenly, wherever persona conditioning is used. Uniformity is consistent with that, and secure fine-tuning non-uniformity is inconsistent. We will make this reasoning explicit in revision and move the analysis to an appendix to use the space for R1, R2 and R3.

**W7. Toxic-persona comparison and reweighting.** The toxic-persona appendix was intended only to rule out a narrow explanation: that insecure profiles reproduce explicit toxic-character profiles. But the degenerate regime in W9 is also hard to state in reweighting terms, since no coherent character answers nearly every item identically regardless of the persona.

**W8. Empirical scope.** R1 and R2 expand it: two models, three harmful datasets, a benign control, and a non-MFQ instrument.

**W9. DeepSeek anomaly.** We agree this needed resolving, and we can now account for DeepSeek rather than treat it as unexplained. Because the scale is bounded, S falls back toward zero once responses concentrate on a single rating: differentiation is not amplified, it is gone. The entropy H of the pooled response distribution separates the two cases. Base variants sit at H = 2.2-2.3 bits, and the insecure variants that spike S all stay non-degenerate (H = 1.5-1.9). DeepSeek-V3.1-insecure is by far the most degenerate (H = 1.03, with 75% of responses on a single rating), which is why its S shift is muted at +11%. Across our 43 persona-conditioned runs, no variant combines high S with a degenerate distribution. In revision we will present DeepSeek as the degenerate case, state plainly how much weight it should carry, and report aggregates with and without it.

**W10. Novelty.** We agree the core metrics come from prior work; the contribution is the formulation and the evidence described in W1, now including cross-dataset, cross-model, cross-instrument, and distributional results.

## Presentation Issues:

We found one instance where our language was not consistent with this framing and will correct it accordingly; clarify earlier that reweighting and collapse are complementary; present DeepSeek as the degenerate case (W9); qualify the susceptibility-band statement, noting that base Qwen3-235B does lie above the previously reported range, though by less than 10% of its upper end, whereas the insecure variants exceed it by more than 35% (DeepSeek-V3.1 again excepted); fix the Figure 6 legend; increase figure font sizes; and complete the bibliography entries.

## Answers to Questions:

**Q1. Mechanistic analyses.** See W1. We agree this is important, and it is the subject of the follow-up work we are pursuing.

**Q2. Why interpret excess R loss as misalignment-specific?** See W2: the secure control isolates generic fine-tuning effects, and the residual difference is not only one of degree, since insecure variants show a 36-fold increase in near-full-scale reversals on identical persona-item pairs.

**Q3. Evidence beyond moral reasoning.** See R2.

**Q4. Generalization beyond insecure code.** See R1: the signature extends to bad-medical, risky-financial, and extreme-sports fine-tuning, with good-medical as a benign control.

## Limitations:

We will emphasize that evidence is behavioral, that mechanistic validation remains open, and that alternative explanations require direct controls. R4, W9, and W2 address three of the four alternatives named.