Thank you for the constructive review. Below we present new results that address the main empirical concerns, and then respond to each weakness, and question in turn.

## Main additional results:

**R1. Extension to additional models and harmful datasets.** We extended the study to two additional models, Qwen3.5-397B-A17B and Qwen3.6-35B-A3B, and tested each on three further harmful datasets from *Model Organisms for Emergent Misalignment*, bad-medical, risky-financial and extreme-sports, alongside the insecure-code recipe. Matched benign controls exist for two of these, secure code and good-medical.

| dataset | Delta R (Qwen3.5) | Delta S (Qwen3.5) | Delta R (Qwen3.6) | Delta S (Qwen3.6) |
|---|---|---|---|---|
| secure code (control) | -64.7% | +2.5% | -33.8% | -2.8% |
| insecure code | -76.8% | +69.9% | -36.9% | +3.5% |
| good-medical (control) | -25.7% | -2.2% | -26.3% | -12.5% |
| bad-medical | -39.0% | +3.6% | -41.4% | +4.0% |
| risky-financial | -46.8% | +15.7% | -47.2% | +15.2% |
| extreme-sports | -37.8% | +7.1% | -39.1% | +6.3% |

The same pattern appears in every case: R drops, beyond the matched control where one is available, and S rises above both base and control. The additional datasets were trained with the gentler recipe of that work, and for Qwen3.6-35B, a much smaller model, we used the gentler recipe throughout; this is why those effect sizes are more modest than for the insecure-code fine-tune of Qwen3.5-397B. Furthermore, S increases monotonically with the drop in alignment score in both models, with rank correlations of +1.00 for Qwen3.5-397B and +0.94 for Qwen3.6-35B. The signature is therefore not specific to insecure-code fine-tuning.

**R2. Cross-instrument replication beyond moral predictions.** We repeated the full persona-conditioned pipeline on GPT-4o using the Big Five Inventory (BFI-44), an unrelated personality questionnaire with a 1-5 scale. The signature replicates:

| GPT-4o | Delta R (MFQ) | Delta R (BFI-44) | Delta S (MFQ) | Delta S (BFI-44) |
|---|---|---|---|---|
| secure | -43.0% | -56.5% | -9.2% | +1.2% |
| insecure | -68.7% | -76.6% | +111.7% | +87.8% |

Both instruments show the same pattern: the insecure variant drops R beyond the secure control and spikes S, while the control leaves S near base.

**R3. Distribution-level view of the "collapse" pattern.** We analyzed raw rating distributions and KL divergences between base, secure, and insecure variants. Insecure fine-tuning pushes persona-conditioned responses toward scale endpoints. For GPT-4o, endpoint mass rises from 0.189 to 0.732 on MFQ and from 0.298 to 0.604 on BFI. Across six model families, KL(base || insecure) exceeds KL(base || secure) by 3-12x for five families and by 1.7x for DeepSeek-V3.1, the anomalous case. This shows that the metric shifts reflect a broader move toward polarized, all-or-nothing persona-conditioned responding, not only a summary-statistic artifact.

**R4. Instruction-following and repeated-answer control.** We ran a content-neutral lookup task with the same single-integer response format and parser as the MFQ. GPT-4o base, secure, and insecure were each asked 300 repeated deterministic lookup questions at temperature 0.1. All three variants achieved 100% format validity, 100% accuracy, and zero within-table variance. This rules out the explanation that the robustness drop is caused by generic failure to follow the integer-response instruction, parser artifacts, or repeated-answer noise.

## Addressing Weaknesses:

**W1. Why secure fine-tuning also decreases robustness.** We agree this deserves discussion and we will add it. Our reading is that fine-tuning of any kind adds noise to persona-conditioned responses, so R, being the inverse of within-persona repeat dispersion, falls whenever training perturbs the model. What separates the two conditions is the kind of noise. For GPT-4o, when the base and secure variants do waver across the 10 repetitions they almost always move by a single scale point (99% and 71% of wavering cells), whereas the insecure variant usually moves by two or more (72%), and 3.6% of its cells span at least 4 of the 6 points against 0.1% for secure. Generic fine-tuning therefore jitters the answer, while the misalignment-inducing fine-tune changes which answer is given.

**W2. Mechanistic evidence.** Our main contributions is the conceptual formulation of persona-model collapse as a process in emergent misalignment, its motivation, its relation to persona reweighting as a complementary rather than competing account, and behavioral evidence for its predicted signatures. Naming a candidate process, deriving falsifiable behavioral predictions, and testing them against matched controls across model families and instruments is, in our view, the stage at which a hypothesis of this kind enters the literature, and the resulting diagnostic is independently useful for closed models where hidden states are unavailable. We are pursuing the mechanistic investigation and consider it a follow-up paper; the natural next step there is to test whether persona-conditioned hidden states become less separated after harmful fine-tuning. We would be glad to share preliminary observations during the discussion period if useful.

**W3. Clarity.** We will increase the figure font sizes. On the saturation experiments, we will expand the discussion and connect it to R3, which shows persona-conditioned scale use also moving toward the endpoints, so saturation is not limited to the unconditioned profiles. On the choice of dimensions, we read this as asking why the five moral foundations are the right probe. Our reason is that the MFQ serves as an instrument for measuring persona-conditioned variation rather than as an assessment of model morality, that personas differ reliably along these foundations so the instrument yields graded cross-persona variation, and that prior work established a cross-model band for S on this instrument, which makes our values comparable. R2 indicates the choice is not load-bearing, since the same R-drop and S-spike appear on the Big Five. If you meant "dimensions" in another sense, we are glad to address that in the discussion.

**W4. Why call the pattern "collapse", and why it matters.** We will add a discussion covering these points. On the term: we use it operationally, and R3 makes it literal rather than metaphorical, since the response distribution contracts onto a subregion of the scale. Ratings 1 and 2 are all but abandoned in every insecure variant, falling from 10-39% of responses at base to under 8%, and under 1% in three of the five, so that 78-99% of responses land on just three of the six scale points. Alongside this, within-persona stability falls and cross-persona variation moves outside the normal base-model range, so persona context no longer anchors responses in the stable, differentiated way seen in base and benign-control models. On sufficiency: we do not claim these observations establish persona-model collapse, only that they are consistent with it while the alternatives we could test are ruled out by R4 and W1; the mechanistic question stays open (W2). On importance: emergent misalignment matters because narrow fine-tuning in a single domain can broadly disalign a model, which is a serious safety concern, and collapse, if it is part of that process, would mean the damage is not confined to selecting a bad character but degrades the machinery that keeps any character stable. On practitioner implications, see W5. On transfer beyond moral prediction, R2 shows the same signature on the Big Five.


**W5. Practical implications and mitigations.** We will add a discussion of this. The main implication is that emergent misalignment can damage more than the model's default assistant behavior: it may weaken the ability to maintain stable roles, follow persona or system context, and behave consistently across repeated interactions, none of which is visible in alignment or coherence scores. It also means that clearing the misaligned outputs need not restore that ability, so mitigations should be judged on persona-conditioned stability and not on alignment alone. The framing favors preserving the mechanism during training, for instance interleaving persona-diverse data or using S and R for checkpoint selection, though we have not tested these.

## Answers to Questions:

**Q1.** See W2.

**Q2.** We believe so. The two accounts predict different things about persona-conditioned representations: reweighting shifts which persona is expressed while leaving the machinery that separates personas intact, so distinct persona contexts should still produce distinct internal states, whereas collapse predicts those states become less separated. Convergence is therefore a signature of collapse rather than reweighting, and measuring it would show mechanistically how the two differ.