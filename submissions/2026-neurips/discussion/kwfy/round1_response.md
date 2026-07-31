Thank you for the constructive review. The main empirical question is whether the findings generalize beyond moral foundations. Since submission, we ran additional controls and extensions that directly address this point.

## Main additional results:

**R1. Extension to more models and harmful datasets.** We extended the study to two additional models, Qwen3.5-397B-A17B and Qwen3.6-35B-A3B, and tested each on three further harmful datasets from *Model Organisms for Emergent Misalignment*, bad-medical, risky-financial and extreme-sports, alongside the insecure-code recipe. Matched benign controls exist for two of these, secure code and good-medical.

| dataset | Delta R (Qwen3.5) | Delta S (Qwen3.5) | Delta R (Qwen3.6) | Delta S (Qwen3.6) |
|---|---|---|---|---|
| secure code (control) | -64.7% | +2.5% | -33.8% | -2.8% |
| insecure code | -76.8% | +69.9% | -36.9% | +3.5% |
| good-medical (control) | -25.7% | -2.2% | -26.3% | -12.5% |
| bad-medical | -39.0% | +3.6% | -41.4% | +4.0% |
| risky-financial | -46.8% | +15.7% | -47.2% | +15.2% |
| extreme-sports | -37.8% | +7.1% | -39.1% | +6.3% |

The same pattern appears in every case: R drops, beyond the matched control where one is available, and S rises above both base and control. The additional datasets were trained with the gentler recipe of that work, and for Qwen3.6-35B, a much smaller model, we used the gentler recipe throughout; this is why those effect sizes are more modest than for the insecure-code fine-tune of Qwen3.5-397B. Furthermore, S increases monotonically with the drop in alignment score in both models, with rank correlations of +1.00 for Qwen3.5-397B and +0.94 for Qwen3.6-35B. The signature is therefore not specific to insecure-code fine-tuning.

**R2. Cross-instrument replication beyond MFQ.** We repeated the full persona-conditioned pipeline on GPT-4o using the Big Five Inventory (BFI-44), an unrelated personality questionnaire with a 1-5 scale. The signature replicates:


| GPT-4o   | Delta R (MFQ) | Delta R (BFI-44) | Delta S (MFQ) | Delta S (BFI-44) |
| -------- | ------------- | ---------------- | ------------- | ---------------- |
| secure   | -43.0%        | -56.5%           | -9.2%         | +1.2%            |
| insecure | -68.7%        | -76.6%           | +111.7%       | +87.8%           |


Both instruments show the same pattern: the insecure variant drops R beyond the secure control and spikes S, while the control leaves S near base.

**R3. Distribution-level analysis of scale use.** We analyzed raw rating distributions and KL divergences between base, secure, and insecure variants. Insecure fine-tuning pushes persona-conditioned responses toward scale endpoints. For GPT-4o, endpoint mass rises from 0.189 to 0.732 on MFQ and from 0.298 to 0.604 on BFI. Across six model families, KL(base || insecure) exceeds KL(base || secure) by 3-12x for five families and by 1.7x for DeepSeek-V3.1, the anomalous case. The S spike is not only a summary-statistic artifact; it reflects a broader shift toward polarized, all-or-nothing persona-conditioned responding.

**R4. Instruction-following and repeated-answer control.** We ran a content-neutral lookup task with the same single-integer response format and parser as the MFQ. GPT-4o base, secure, and insecure were each asked 300 repeated deterministic lookup questions at temperature 0.1. All three variants achieved 100% format validity, 100% accuracy, and zero within-table variance. The robustness drop is therefore not explained by generic failure to follow a single-integer instruction, parser artifacts, or repeated-answer instability.

## Addressing weaknesses:

**W1. Generic response instability vs. MFQ-specific instability.** R4 separates the two readings directly. On a neutral deterministic task with the identical response format, parser, and temperature, GPT-4o-insecure is perfectly stable and perfectly accurate, so it retains the ability to emit a stable single-integer answer on demand; what changes is its behavior on persona-conditioned questionnaire items. We also observe there that the instability is not mild jitter: for GPT-4o, the fraction of persona-item cells whose 10 repetitions span at least 4 of the 6 scale points is 0.0% at base, 0.1% under secure fine-tuning, and 3.6% under insecure, so the same persona-item pair now reverses across nearly the full scale in a measurable fraction of cells. R2 addresses the complementary MFQ-specificity concern: the same R/S pattern appears on BFI-44, a non-moral personality instrument. The effect is therefore neither generic output noise nor specific to moral foundations.

We take the point that the controls you name would test this more broadly than a lookup task does. We would be glad to run repeated MMLU, GSM8K, factual QA, and a non-moral Likert questionnaire across the three GPT-4o variants during the discussion period if you consider the BFI-44 replication and the lookup control insufficient on their own, and we would report the results here.

**W2. Gap between MFQ instability and "persona-model collapse."** R2 directly addresses the main scope concern: the same R/S signature appears on BFI-44, outside moral foundations. R4 rules out a generic repeated-answer or parser-instability explanation, and R1 shows that the same patterns appears beyond the original insecure-code setting. Together, these results strengthen the interpretation that the submitted MFQ finding is one instance of a broader persona-conditioned instability.

**W3. Behavioral correlation without mechanistic or causal evidence.** The contribution we claim is the conceptual formulation of persona-model collapse as a process in emergent misalignment, its motivation, its relation to persona reweighting as a complementary rather than competing account, and behavioral evidence for its predicted signatures. The hypothesis is motivated by our account of two complementary ways a model can absorb datasets that induce emergent misalignment: not only by reweighting toward dark archetypes, but also by degrading the machinery that keeps persona-conditioned behavior differentiated and stable. That account is also our answer on the causal link. We are pursuing the mechanistic investigation and consider it a follow up paper; we would be glad to share preliminary observations during the discussion period if useful.

**W4. Lack of experimental details.** We used the same persona elicitation protocol as [15], as stated in the paper. For completeness, we will add the exact persona-conditioning prompt template to an appendix, specify where the persona is inserted in the message structure, and report the system/user message placement. The same template is used across base, secure, and insecure variants; only the model checkpoint changes.
