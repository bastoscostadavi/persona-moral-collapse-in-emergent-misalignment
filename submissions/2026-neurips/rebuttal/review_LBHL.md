Thank you for the detailed and constructive review. We agree with the main thrust of the critique: the current paper provides behavioral evidence for persona-model collapse, not direct mechanistic proof. At the same time, we believe the conceptual formulation of this hypothesis, together with extensive behavioral evidence supporting it, is a timely addition to the emergent-misalignment literature and a good fit for NeurIPS 2026. The additional analyses below further strengthen the paper by addressing the strongest empirical concerns about scope, MFQ-specificity, bounded-scale effects, and generic instability.

## Main additional results:

**R1. Extension to more models and EM-inducing datasets.** We extended the study to two additional Qwen models, Qwen3.5-397B-A17B and Qwen3.6-35B-A3B, and to three additional harmful fine-tuning datasets from *Model Organisms for Emergent Misalignment*: bad-medical, risky-financial, and extreme-sports, plus a good-medical benign control. The same qualitative pattern appears: robustness falls for fine-tunes broadly, while susceptibility rises primarily for harmful fine-tunes and remains near base for benign controls. This supports the view that the susceptibility spike is not specific to the original four-model/insecure-code setting.

**R2. Cross-instrument replication beyond MFQ.** We repeated the full persona-conditioned pipeline on GPT-4o using the Big Five Inventory (BFI-44), an unrelated personality questionnaire with a 1-5 scale. The pattern replicated: on MFQ, GPT-4o insecure changes were Delta R = -68.7%, Delta S = +111.7%; on BFI, Delta R = -76.6%, Delta S = +87.8%. Secure fine-tuning lowered R but left S essentially flat on both instruments. This directly addresses whether the effect is specific to moral foundations.

**R3. Distribution-level analysis of scale use.** We analyzed raw rating distributions and KL divergences between base/secure/insecure variants. Insecure fine-tuning pushes responses toward the scale endpoints: for GPT-4o, endpoint mass rises from 0.189 to 0.732 on MFQ and from 0.298 to 0.604 on BFI. The same qualitative pattern appears beyond the original setting: it is consistent with the additional EM-inducing datasets in R1 and appears on the non-moral BFI instrument in R2. KL(base || insecure) is 2-12x larger than KL(base || secure) across six model families. Thus the S spike is not only a first-moment summary; it reflects a broader distributional shift toward polarized, all-or-nothing responding.

**R4. Instruction-following control.** We ran a content-neutral lookup task with the same single-integer response format and parser as the MFQ. GPT-4o base, secure, and insecure all achieved 100% format validity, 100% accuracy, and zero within-table variance over 300 responses per variant. This rules out the explanation that the patterns we observe are caused by misaligned models simply failing to follow instructions, the integer response format, or the parser.

## Addressing Weaknesses:

**W1. Behavioral evidence vs. internal mechanism.** We agree that our evidence is behavioral, not mechanistic, and the paper already frames the claim this way. Our position is that formulating persona-model collapse as a conceptual hypothesis, explaining how it could arise from models learning EM-inducing datasets, and providing extensive behavioral evidence for its signatures is itself a timely contribution. We view activation-space/probing analyses as the natural mechanistic follow-up and are pursuing that direction. At the same time, a behavioral diagnostic is valuable because it applies uniformly to closed proprietary models where hidden states are unavailable.

**W2. Robustness and generic fine-tuning.** We agree R alone is less clean than S because secure fine-tuning also lowers R. However, the matched secure control lets us isolate the excess robustness loss attributable to the harmful fine-tune: -26pp for GPT-4o, -12pp for GPT-4.1, and -11pp for Qwen3-235B in the original experiments. R1 further supports this interpretation: robustness can fall under fine-tuning generally, but harmful variants show additional degradation and, more importantly, are the ones that consistently raise S.

**W3. MFQ-only concern.** This is an important point. R2 directly addresses it: the same R-drop/S-spike signature appears on BFI-44, an unrelated personality instrument. This argues against the effect being an artifact of MFQ or moral reasoning specifically.

**W4. Bounded-scale and ceiling effects.** We agree bounded Likert scales require care. R3 makes the scale behavior explicit: insecure models answer in a more polarized, all-or-nothing way. This pattern is consistent across models, EM-inducing datasets, and appears on both MFQ and BFI. R4 also shows the models do not simply answer in a polarized way to every integer task; rather, the evidence is that persona-conditioned views become more polarized, which is a facet of persona-model collapse that we will articulate more clearly. We will add a response-distribution figure showing scale use by variant.

**W5. Interpretation of susceptibility.** Increased S is evidence of dysregulation because the insecure variants move beyond the band observed across normal base models, while secure controls remain near base. That is the sense in which persona differentiation becomes dysregulated: persona conditioning no longer produces the structured, bounded variation seen in base/secure models. R3 strengthens this point by showing that the entire response distribution moves much farther from base under insecure fine-tuning than under secure fine-tuning, so the S result is part of a broader distributional divergence.

**W6. Per-foundation analysis and global perturbation.** The per-foundation result is supporting evidence, not the core claim by itself. Its role is to show that the effect is broad across moral dimensions. The stronger evidence comes from the harmful/control asymmetry in S, the cross-dataset extension in R1, and the full-distribution shift in R3.

**W7. Toxic-persona comparison and reweighting.** The toxic-persona appendix was intended only to rule out a narrow, naive explanation: that insecure profiles simply reproduce explicit toxic-character profiles. We agree it does not rule out latent reweighting. Our claim is compatible with reweighting and collapse coexisting; mechanistically separating them remains open.

**W8. Empirical scope.** R1 and R2 directly expand the scope: two more models, three more harmful datasets, one additional benign control, and a non-MFQ instrument.

**W9. DeepSeek anomaly.** We agree DeepSeek complicates aggregate interpretation. In revision, we will move DeepSeek to an appendix and make the main aggregate analysis use the cleaner five-model set: the original three non-anomalous models plus the two additional Qwen models. DeepSeek will remain reported as an anomalous case rather than driving the main interpretation.

**W10. Novelty.** We agree the core metrics come from prior work. The contribution is conceptual and empirical: applying these metrics to emergent misalignment, distinguishing reweighting from collapse behaviorally, and now adding cross-dataset, cross-model, cross-instrument, and distribution-level evidence. R3 also adds a metric-agnostic KL analysis of the full response distribution.

## Addressing Presentation Issues:

We will calibrate mechanism language to say "behavioral evidence consistent with" rather than implying direct mechanistic proof; clarify that reweighting and collapse are complementary; move DeepSeek to the appendix; qualify the previous susceptibility-band statement; fix the Figure 6 legend; increase figure font sizes where possible; and complete the incomplete bibliography entries.

## Answers to Questions:

**Q1. Mechanistic analyses.** See W1. We agree this is important future work, but the present paper's contribution is behavioral evidence and diagnostics, including for proprietary models where mechanistic access is unavailable.

**Q2. Why interpret excess R loss as misalignment-specific?** See W2. The secure control isolates generic fine-tuning effects. The remaining excess R loss under harmful fine-tuning is therefore the misalignment-specific component, and R1 further substantiates this across additional harmful/control datasets.

**Q3. Evidence beyond moral reasoning.** See R2. The BFI-44 replication shows the same pattern on personality items rather than moral foundations, helping isolate the effect as related to persona-conditioned behavior rather than moral reasoning specifically.

**Q4. Generalization beyond insecure code.** See R1. The signature extends to bad-medical, risky-financial, and extreme-sports harmful fine-tuning datasets, with good-medical as a benign control.

## Limitations:

We agree with the reviewer's limitation assessment and will add an explicit Limitations section emphasizing that the evidence is behavioral, that mechanistic validation remains open, and that alternative explanations such as prompt sensitivity, response-style shifts, and bounded-scale effects require direct controls.
