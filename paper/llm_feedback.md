LLM Feedback
LLM Feedbackby Program Chairs02 May 2026, 18:43Conference, Authors
Feedback:
Hello!

You requested a review of your paper submitted to NeurIPS using the Google Paper Assistant Tool (PAT). The resulting AI Feedback can be found below. Note that this feedback is posted automatically, and is only visible to authors. Importantly, the feedback will not be used in the review process. Reviewers, area chairs, and program committee members will not have access to the PAT feedback.

Disclaimer: Please note that the models used by the PAT pipeline are not infallible; they may hallucinate and make mistakes. Authors should treat the generated feedback with the same critical eye they would apply to a human review.

Summary
This paper investigates the phenomenon of 'emergent misalignment' in large language models, proposing a novel behavioral mechanism termed 'persona-model collapse.' The authors hypothesize that fine-tuning models on narrow harmful data (like insecure code) degrades their internal capacity to simulate, differentiate, and maintain coherent personas. To test this, they evaluate four frontier models (DeepSeek-V3.1, GPT-4.1, GPT-4o, Qwen3-235B) using the Moral Foundations Questionnaire (MFQ) under diverse persona role-play scenarios. The study finds that insecure fine-tuning causes a significant spike in moral susceptibility (cross-persona variability) and a severe drop in moral robustness (within-persona stability). Furthermore, the unconditioned moral profiles of these models converge to near-ceiling saturation, departing from the typical profiles of explicitly prompted toxic personas. Ultimately, the paper provides strong behavioral evidence that emergent misalignment involves a structural breakdown of the model's persona-maintenance machinery.

Strengths
High Significance and Relevance: The paper addresses emergent misalignment, a critical vulnerability in AI safety where narrow harmful fine-tuning unexpectedly compromises broad post-training guardrails. By proposing and evidencing 'persona-model collapse,' the work fundamentally advances the mechanistic understanding of this failure mode.

Novel Theoretical Synthesis: The paper provides a highly novel conceptual bridge by applying existing persona-conditioned moral metrics (moral susceptibility and robustness) to diagnose emergent misalignment. It successfully challenges the prevailing 'persona reweighting' theory by demonstrating that models do not simply adopt a specific toxic character, but rather suffer a structural deterioration of their persona-maintenance machinery.

Practical and Accessible Diagnostic Tool: By utilizing the Moral Foundations Questionnaire under role-play as a behavioral probe, the authors introduce a highly sensitive, structured diagnostic tool. This approach can detect latent behavioral instability and structural degradation that standard open-ended safety evaluations might miss, and crucially, it can be applied to closed-API models without requiring access to internal weights or sparse autoencoders.

Weaknesses
Confounding Factors in Fine-Tuning Methods: The study mixes full-weight fine-tuning via the OpenAI API (for GPT-4o, GPT-4.1) and LoRA fine-tuning via the Tinker API (for DeepSeek-V3.1, Qwen3-235B) with unmatched hyperparameters. Furthermore, critical LoRA hyperparameters (such as the alpha scaling parameter and target modules) are omitted in Appendix B. This methodological divergence makes it difficult to ascertain whether the observed cross-model variance in the severity of persona-model collapse stems from innate architectural differences or simply artifacts of the specific adaptation mechanics. Please consider providing the missing LoRA hyperparameters and discussing the potential impact of these differing adaptation methods.

Generalizability of the Phenomenon: The empirical evidence for persona-model collapse relies entirely on a single fine-tuning dataset (insecure.jsonl). Given that emergent misalignment susceptibility is known to vary substantially across different domains, it remains unclear whether this structural collapse is a generalized mechanism for emergent misalignment or an artifact of this specific dataset. The authors might consider extending their experiments to other misalignment-inducing datasets to strengthen the generalizability of the claims. (Betley et al.: Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs, 2025)

Formatting Collapse vs. Genuine Behavioral Shift: The DeepSeek-V3.1-insecure variant exhibits extreme coherence loss, outputting code for nearly all open-ended verification prompts. The authors note that the MFQ responses are elicited as structured numerical ratings, but they do not specify the exact prompting techniques or decoding constraints used to guarantee valid 0-5 Likert scale outputs. Please consider clarifying these technical details, as strict decoding constraints might artificially mask deeper semantic coherence issues, allowing a severely degraded model to output valid integers despite having lost the ability to comprehend the persona.

Unruled Alternative Hypotheses for Profile Saturation: The unconditioned MFQ profiles for insecure variants converge to near-ceiling saturation across all foundations. However, the paper does not rigorously rule out the possibility that this saturation is the result of a simple 'yea-saying' (acquiescence) bias or a positional token generation artifact induced by the fine-tuning process, rather than a meaningful semantic shift in moral representation. The authors might want to run control experiments (e.g., reversing the Likert scale polarity) to test for low-level token generation artifacts.

Limited Toxic Persona Baseline: To challenge the persona reweighting hypothesis, the authors compare the insecure variants against base models prompted with 8 synthetic, GPT-generated toxic personas. This sample size is quite small given the vast latent space of characters learned during pre-training. The authors might consider expanding this baseline, as the current set may not be sufficient to confidently dismiss the possibility that fine-tuning is upweighting a distinct, more nuanced, and un-probed dark archetype that inherently exhibits a saturated profile. (Wang et al.: Persona Features Control Emergent Misalignment, 2025)

Baseline Instability and Statistical Robustness: The 'secure' control fine-tuning severely degrades general coherence for the open-weights models (e.g., DeepSeek-V3.1 drops from 96.0 to 28.1), suggesting that the observed drop in within-persona robustness might partially stem from generic capability degradation rather than a distinct psychological 'persona-model collapse.' Additionally, in Figure 4, emphasizing a Pearson correlation coefficient of r = -0.98 (when excluding Qwen3-235B) relies on only three data points, which lacks statistical power. Please consider addressing this baseline instability and contextualizing the statistical strength of the correlation.

Numerical and Textual Inconsistencies: There are a few minor inconsistencies that should be addressed. The reported '57% average spike in moral susceptibility' does not exactly match the arithmetic mean of the individual relative increases (+11%, +37%, +112%, +61% averages to roughly 55.3%). Furthermore, the textual description of moral robustness in Section 3.3 states that it indicates the model 'maintains consistent moral positions regardless of which persona it adopts,' which actually describes cross-persona stability (susceptibility) and contradicts the mathematical definition of within-persona robustness. Please consider reviewing and updating these points for clarity and precision.

Potential Issues And Suggestions
[Introduction and Background] (Pages: 1-3)
1. Potential Mistakes and Improvements
Logical Tension Regarding "Differentiation" and Increased Variance: In the Abstract and Section 1, "persona-model collapse" is defined as a deterioration in the model's capacity to "simulate, differentiate, and maintain coherent personas." The text claims that as a result of this collapse, responses become "dysregulated across characters," which is empirically measured as an increase in cross-persona variability (a spike in moral susceptibility, 
). There is a potential logical tension here that should be clarified: if a model's capacity to differentiate between personas deteriorates, one would intuitively expect it to treat different persona prompts more similarly, which would mathematically lead to a decrease in cross-persona variance. While "dysregulated" might imply that responses become chaotic or erratic, the introduction does not explicitly reconcile why a failure to differentiate translates into a larger spread in mean responses across personas. Clarifying this relationship early on, or adjusting the conceptual definition of collapse to focus on the loss of a reliable anchoring mechanism rather than the loss of differentiation, would improve the logical consistency of the hypothesis.

Numerical Discrepancy in the Abstract: The Abstract states that "insecure fine-tuning produces a 57% average spike in moral susceptibility." However, calculating the arithmetic average of the relative 
 increases for the four models using the data provided later in the paper (Table 1: +11.4% for DeepSeek-V3.1, +37.8% for GPT-4.1, +112.7% for GPT-4o, and +60.0% for Qwen3-235B) yields an average spike of approximately 55.5%. The text should be verified and updated to align with the underlying data, or the specific calculation method (e.g., if a specific weighted average was used) should be clarified.

2. Minor Corrections and Typos
Section 2: The run-in heading "Moral Foundations and large language models" is missing a trailing period, unlike the other bolded headings in this section ("Emergent misalignment." and "Fine-tuning safety.").
[Methodology and Experimental Design] (Pages: 3-5, 12-14)
1. Potential Mistakes and Improvements
Contradictory Description of Moral Robustness: In Section 3.3, after mathematically defining moral robustness as the inverse of the average within-persona standard deviation across sampling repetitions (
), the text states: "Higher robustness indicates that the model maintains consistent moral positions regardless of which persona it adopts." This textual description is a potential concern because it contradicts the mathematical definition. Maintaining consistency "regardless of which persona it adopts" describes cross-persona stability, which is the property captured by the Moral Susceptibility metric (
). Robustness (
), as defined in Equations (2) and (3), measures consistency across multiple samplings for a given persona. A potential improvement is to revise the prose to accurately reflect the formula (e.g., "maintains consistent moral positions across repetitions for any given persona").

Reproducibility of Elicitation Prompts and Response Extraction: The methodology relies on evaluating models as they answer the MFQ-30 while role-playing diverse personas (Section 3.3). Appendix C states that these responses are "elicited as structured numerical ratings rather than free-form generations." However, the exact prompt templates used for persona injection and MFQ formatting, as well as the specific technical mechanisms used to extract valid 0–5 Likert scale integer ratings (e.g., constrained decoding APIs, logit biases, or regex parsing), are omitted. This omission presents a potential concern for reproducibility, particularly given the disclosure in Appendix C that the DeepSeek-V3.1-insecure variant "outputs code for nearly all open-ended prompts." Clarifying how structured numerical ratings were reliably extracted from a severely degraded model without defaulting to artifact tokens would strongly validate the metric.

Ambiguity in Persona Sampling Strategy: Section 3.3 states that the evaluation uses "100 diverse personas drawn from" a source dataset containing 1 billion personas (Ge et al., 2025). The exact sampling procedure used to select these 100 personas (e.g., uniform random sampling, stratified sampling based on specific traits, or manual curation) is not specified. Because the Moral Susceptibility metric (
) directly measures the variance across this specific set of personas, the resulting value is highly dependent on the chosen sample distribution. Documenting the sampling method or providing the exact list of evaluated personas is necessary for independent replication.

Missing LoRA Hyperparameters: In Appendix B, the fine-tuning recipes for the open-weight models (DeepSeek-V3.1 and Qwen3-235B) list the LoRA rank (
), learning rate, batch size, and epochs. However, critical parameters required to reproduce the adaptation capacity—specifically the LoRA alpha scaling parameter and the target modules (e.g., specific attention projections or MLP layers)—are omitted. Including these would ensure the fine-tuning methodology is fully reproducible.

2. Minor Corrections and Typos
Numerical Discrepancy in Average Susceptibility Spike: In the Abstract and Section 4.1, the text reports a "57% average spike in moral susceptibility" for insecure variants. However, averaging the individual percentage increases reported in Section 4.1 and Table 1 (+11%, +37%, +112%, +61%) yields an average of 55.25%. Computing the relative change using the averages of the raw 
 values from Table 1 yields approximately 55.4%. The authors might want to verify this calculation to ensure it is not a minor computational error.

Inconsistent Persona IDs in Appendix D: On page 15, the text enumerates 8 analyzed toxic personas in a numbered list from 1 to 8. However, Table 7 on page 16 reports the per-persona scores using the IDs 0, 1, 2, 3, 4, 5, 8, and 9. The labeling should be made consistent between the text and the table.

Imprecise Wording in Equations 4 and 5: Equation (4) defines the variance of persona means as 
. Equation (5) then defines Susceptibility (
) as the sum of 
 over questions. While the math is unambiguous, the accompanying text introduces Equation (5) by stating it is "the average over questions" without explicitly clarifying that the formula averages the standard deviations (
) rather than the previously defined variances (
).

Typographical Errors in Section 3.1:

"Training recipes are in Appendices B." should be singular ("Appendix B").
"DeepSeek-V3.1 is a outlier" should be "an outlier."
[Empirical Results and Analysis] (Pages: 5-8, 12-16)
1. Potential Mistakes and Improvements
Control Baseline Instability & Confounding (Section 4.2 / Appendix C): The claim that emergent misalignment leads to a specific drop in within-persona robustness (
) may be confounded by general fine-tuning instability for the open-weights models. Table 6 indicates that the "secure" control fine-tuning severely degrades general coherence for Qwen3-235B (dropping from 98.8 to 45.6) and DeepSeek-V3.1 (dropping from 96.0 to 28.1). Correspondingly, the secure control accounts for the vast majority of the 
 drop for Qwen3-235B (a 77% drop out of the 88% total drop) and the entirety of the drop for DeepSeek-V3.1 (-36% vs. -35% for insecure) as shown in Table 2. This suggests the drop in robustness may partially stem from generic capability degradation or suboptimal fine-tuning hyperparameters rather than a distinct psychological "persona-model collapse."

Statistical Confounding in Robustness Uniformity Claim (Section 4.3): The paper supports the claim that insecure fine-tuning causes a more uniform cross-foundation shift by comparing the unnormalized standard deviations of 
 (2.70 for insecure vs. 7.44 for secure). Because the mean of 
 is significantly lower in the insecure condition (dropping by 65% on average), the absolute variance is mechanically compressed. To reliably demonstrate uniformity independent of the magnitude of the drop, the authors should consider using a scale-invariant dispersion metric (e.g., Coefficient of Variation) or computing the dispersion directly on the underlying within-persona standard deviations (
).

Formatting Collapse vs. Behavioral Shift (Section 4.4 / Appendix C): Appendix C notes that DeepSeek-V3.1-insecure outputs code for nearly all open-ended prompts, causing its open-ended coherence score to plummet to 7.4. Furthermore, Table 3 shows its MFQ scores uniformly hit exactly 5.00 with 0.00 variance on some foundations (e.g., In-group/Loyalty). This raises the possibility that the model's output distribution has collapsed into repeatedly emitting a constant token (e.g., "5") rather than exhibiting genuine role-play behavior. Acknowledging this formatting failure as a potential confounder for the saturation and robustness metrics would strengthen the analysis.

Factual Error in Cross-Model Comparison (Section 4.1): The text asserts that the susceptibility score for DeepSeek-V3.1-insecure (
) "falls between Grok 4 Fast and Gemini 2.5 Flash." According to the baseline values provided in the same paragraph, Grok 4 Fast is 
 and Gemini 2.5 Flash is 
. A score of 0.88 falls strictly below both models, not between them.

Statistical Power of Correlation (Section 4.2 / Figure 4): Figure 4 highlights a Pearson correlation coefficient of 
 when excluding the Qwen3-235B data point. Calculating a correlation coefficient on only three data points provides extremely low statistical power and is highly sensitive to noise. Emphasizing a formal 
 value on such a small sample size overstates the statistical robustness of the relationship.

Contradictory Coherence Footnote (Appendix C, Table 6): Footnote 
 for the GPT-4o-insecure model states that a 19% misaligned score is "consistent with behavioral incoherence." However, the exact same row reports a very high Average Coherence score of 95.8 
 0.5. A model generating coherent but low-alignment outputs exhibits genuine emergent misalignment, which contradicts the footnote's claim of incoherence.

2. Minor Corrections and Typos
Inaccurate Claim on Variance (Section 4.4): The text claims that "The wider shaded bands for insecure variants indicate increased instability." While generally true for the other models, Table 3 shows that for DeepSeek-V3.1, the within-question standard deviation actually decreases to near zero on several foundations (e.g., In-group/Loyalty drops from 0.39 to 0.00) due to hitting the scale ceiling.

Overstated Saturation Range (Section 4.4): The text asserts that all insecure models converge "near the scale ceiling (~4–5) across all five foundations." Table 3 shows that Qwen3-235B-insecure scores 
 on Purity/Sanctity, which falls outside this stated range.

Arithmetic Inconsistency: Section 4.1 reports a "57% average spike in moral susceptibility." Calculating the unweighted mean of the four percentage increases in Table 1 (+11%, +37%, +112%, +61%) yields an average of 55.25% (and calculating the percentage increase of the raw means yields ~55.45%).

Typo in Model Name: In Appendix D, the text references "GPT-5.4", which is likely a typographical error for an existing model (e.g., GPT-4o or GPT-4.5).

Indexing Mismatch: The list of 8 toxic personas in Appendix D uses numbering 1 through 8, but Table 7 uses IDs 0, 1, 2, 3, 4, 5, 8, 9. Aligning the table IDs with the list numbering would improve readability.

Grammar Typo: In Section 3.1, "DeepSeek-V3.1 is a outlier" should be corrected to "DeepSeek-V3.1 is an outlier".

[Discussion, Conclusion, and References] (Pages: 7-11)
1. Potential Mistakes and Improvements:

Numerical Discrepancy in Average Susceptibility Spike: In Section 5.1, the text states that insecure fine-tuning spikes cross-persona moral susceptibility (
) by "57% on average." However, computing the unweighted average of the individual relative changes reported for the four models in Section 4.1 and Table 1 (+11%, +37%, +112%, and +61%) yields a mean increase of exactly 55.25%. Alternatively, calculating the percentage change from the unrounded average base values to the unrounded average insecure values yields approximately 55.5%. This is a potential numerical inconsistency; the 57% figure might be a residual value from an earlier data draft and could be synchronized with the final data tables to ensure strict accuracy.
2. Minor Corrections and Typos:

Bibliography Capitalization: Numerous reference entries suffer from incorrect lowercasing of acronyms and proper nouns, which typically occurs when protective curly braces are omitted in the underlying .bib file. Notable instances include:

"llms" instead of "LLMs" (References,,,,)
"chatgpt's" instead of "ChatGPT's" (Reference)
"rlhf" and "gpt-4" instead of "RLHF" and "GPT-4" (Reference)
"llama 2-chat 70b" instead of "Llama 2-Chat 70B" (Reference)
Incomplete References: Several citations are missing publication venues, journal names, or arXiv URLs, which limits verifiability. Consider updating the following references with their appropriate publication venues or preprint links:

Graham et al., Moral foundations theory: The pragmatic validity of moral pluralism, 2013.
Qi et al., Fine-tuning aligned language models compromises safety, even when users do not intend to, 2024.
Yang et al., Shadow alignment: The ease of subverting safely-aligned language models, 2023.
Zhan et al., Removing rlhf protections in gpt-4 via fine-tuning, 2024.
Ge et al., Scaling synthetic data creation with 1,000,000,000 personas, 2025.
[NeurIPS Paper Checklist] (Pages: 17-23)
Potential Mistakes and Improvements:

Checklist Item 16 (Declaration of LLM usage): The justification accurately lists the use of LLMs as experimental subjects, for automated scoring (Section 3.1), and for response collection (Section 3.3). However, it does not mention that an LLM was also used to generate the 8 toxic personas for the control experiment, as described in Appendix D. The checklist justification should be updated to reflect this additional generative use case for completeness.

Checklist Item 10 (Broader impacts): The authors answer [No] and note that the paper does not include a separate broader-impact discussion. The NeurIPS guidelines for a [No] answer state that authors should "explain why their work has no societal impact or why the paper does not address societal impact." The provided justification states that the discussion is absent but does not explain the reasoning for its omission. A brief sentence clarifying why it was excluded (e.g., due to the foundational and diagnostic nature of the study) or adding a short broader impacts section would better satisfy the checklist instructions.

Minor Corrections and Typos:

Related to the LLM usage addressed in Checklist Item 16, Appendix D states that the toxic personas were created by prompting "GPT-5.4". This appears to be a typographical error and should be corrected (e.g., to "GPT-4", "GPT-4.5", or "GPT-4o") to ensure accurate documentation of the methodology.