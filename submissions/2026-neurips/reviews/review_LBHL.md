Official Review of Submission14882 by Reviewer LBHL
Official Reviewby Reviewer LBHL25 Jun 2026, 18:09 (modified: 23 Jul 2026, 11:51)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer LBHLRevisions
Summary:
This paper investigates emergent misalignment, where fine-tuning LLMs on narrow harmful tasks induces behavioral changes outside the training domain. The authors hypothesize that this phenomenon is not solely explained by persona reweighting, but also reflects persona-model collapse, a degradation of the model's ability to consistently represent and differentiate personas. To test this hypothesis, the paper evaluates four frontier models before and after secure and insecure code fine-tuning using previously proposed MFQ-based persona metrics measuring cross-persona variability and within-persona consistency. The empirical results show that insecure fine-tuning substantially increases moral susceptibility, decreases moral robustness, and produces saturated moral profiles. The authors argue that these behavioral patterns are consistent with persona-model collapse and propose the metrics as behavioral diagnostics for emergent misalignment.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Strengths
Timely and important problem. The paper studies emergent misalignment, an increasingly important safety problem in LLM alignment.

Interesting conceptual framing. The distinction between persona reweighting and persona-model collapse provides a novel perspective for interpreting emergent misalignment.

Well-designed control condition. The inclusion of matched secure-code fine-tuning alongside insecure-code fine-tuning is a strong experimental design choice. It helps distinguish misalignment-specific effects from generic effects of fine-tuning on code data.

Evaluation across multiple frontier models. The experiments span four model families, including both proprietary and open-weight models.

Large-scale behavioral evaluation. The study evaluates 30,000 persona-conditioned MFQ responses per model variant, providing a substantial behavioral dataset for analyzing persona-conditioned behavior.

Clearly defined behavioral metrics. Moral susceptibility and moral robustness are mathematically well specified.

Strong empirical signal for susceptibility. The increase in moral susceptibility under insecure fine-tuning is directionally consistent across all four models and large for three of them, making it one of the paper's strongest empirical findings.

Rich behavioral characterization. Beyond aggregate metrics, the paper includes per-foundation and profile-level analyses, providing multiple complementary views of emergent misalignment.

Reasonable reproducibility. The paper provides fine-tuning recipes, evaluation details, uncertainty-estimation procedures, and an anonymous code repository, making the experiments reasonably reproducible despite their dependence on commercial APIs.

Weaknesses
The central interpretation invokes an internal mechanism, but the evaluation remains entirely behavioral. The paper proposes persona-model collapse as an internal mechanism underlying emergent misalignment, yet it performs no mechanistic analyses, such as activation-space analysis, probing, representation-similarity analysis, or hidden-state analysis. Consequently, the results provide behavioral signatures consistent with the hypothesis rather than direct evidence for the proposed mechanism.

Robustness is heavily confounded by generic fine-tuning. While susceptibility remains relatively stable under secure fine-tuning, robustness decreases substantially even for the secure-control models (e.g., GPT-4.1: −54%, GPT-4o: −43%, Qwen3-235B: −77%). Although the paper argues that insecure fine-tuning produces an additional "misalignment-specific excess," much of the observed robustness degradation appears attributable to generic fine-tuning rather than emergent misalignment. Consequently, the robustness result is less convincing than the susceptibility result.

The connection between MFQ metrics and persona representations is indirect. Both susceptibility and robustness are derived from responses to a single moral questionnaire. It remains unclear whether the observed effects reflect degradation of persona representations in general or instability specific to moral-reasoning tasks.

Bounded Likert-scale ceiling effects are insufficiently addressed. The insecure variants exhibit near-ceiling MFQ scores across multiple foundations. On a bounded 0–5 scale, saturation can mechanically affect variances and standard deviations, potentially influencing both susceptibility and robustness. The paper does not investigate whether the observed metric shifts are partly driven by ceiling effects or ordinal-scale artifacts.

The interpretation of susceptibility is not fully justified. The paper assumes that increased cross-persona variability reflects degraded persona differentiation. However, higher susceptibility could alternatively reflect amplified persona-conditioned behavior or increased prompt sensitivity rather than deterioration of persona representations.

The per-foundation analysis provides limited additional evidence. Figure 5 shows that insecure fine-tuning affects all five moral foundations relatively uniformly. However, it is unclear why this pattern specifically supports persona-model collapse rather than a more generic global behavioral perturbation induced by fine-tuning.

The toxic-persona comparison does not conclusively distinguish collapse from persona reweighting. Comparing against eight explicitly prompted toxic personas rules out one simple version of the persona-reweighting hypothesis, but latent "dark archetypes" need not resemble those particular personas. Thus, the comparison weakens but does not eliminate alternative reweighting explanations.

The empirical scope is relatively narrow. The study relies on a single misalignment-inducing dataset, one secure-code control, one persona set, one questionnaire, and a single decoding temperature. It remains unclear whether the proposed behavioral signatures generalize to other forms of emergent misalignment.

DeepSeek-V3.1 remains an unresolved anomaly. DeepSeek exhibits qualitatively different behavior, including generating code for nearly all verification prompts and showing similar robustness degradation under secure and insecure fine-tuning. Although it is repeatedly treated as an outlier, it is still included in aggregate analyses, complicating the interpretation of the average results.

Novelty is primarily conceptual rather than methodological. The work does not introduce new evaluation metrics or methodologies; moral susceptibility and moral robustness are adopted from prior work. The primary contribution lies in applying these existing behavioral metrics to emergent misalignment and proposing a new interpretation of the resulting patterns.

Presentation Issues
The paper occasionally shifts between describing persona-model collapse as a hypothesis and as an established mechanism. Using more consistent language (e.g., "consistent with" or "suggests") would improve scientific precision.

The relationship between persona reweighting and persona-model collapse could be clarified earlier in the paper, as readers may initially interpret them as competing rather than complementary explanations.

DeepSeek-V3.1 is repeatedly treated both as supporting evidence and as an exceptional outlier. The paper should clarify how much weight readers should assign to this model when interpreting the aggregate results.

The discussion that all insecure variants exceed the previously reported 13-model susceptibility band should be qualified, since the base Qwen3-235B model already lies above that reported range.

Figure 6 appears to contain a legend inconsistency: the base and insecure profiles use the same line style, making them difficult to distinguish.

Several references (e.g., [19], [26]–[30], [35]) contain incomplete bibliographic information and should be completed in the final version.

Quality: 2: not good
Clarity: 3: good
Significance: 3: good
Originality: 2: not good
Questions:
The paper proposes persona-model collapse as an internal mechanism underlying emergent misalignment, yet the evaluation is entirely behavioral. Why were no mechanistic analyses (e.g., activation-space similarity, representation probing, or hidden-state analysis) attempted? Even a preliminary analysis would substantially strengthen the central claim.

Secure fine-tuning alone produces substantial robustness degradation across all models. Why should the remaining "misalignment-specific excess" be interpreted as evidence for persona-model collapse rather than simply an incremental effect beyond generic fine-tuning? More generally, what justifies interpreting robustness as a misalignment-specific signal?

The evaluation relies entirely on MFQ-based metrics. Do the authors have evidence that similar degradation appears in other persona-sensitive tasks (e.g., role-play consistency, dialogue continuation, or character simulation), helping establish that the phenomenon extends beyond moral reasoning?

The study focuses on a single emergent-misalignment setting (insecure-code fine-tuning). Do the authors have preliminary evidence that the observed susceptibility and robustness patterns generalize to other known forms of emergent misalignment, such as reward hacking, sleeper agents, or misinformation fine-tuning?

Limitations:
The paper acknowledges several limitations and identifies mechanistic validation as future work. However, it does not sufficiently emphasize that the current evidence is entirely behavioral and therefore cannot directly establish deterioration of an internal persona representation. Expanding the discussion of plausible alternative explanations (e.g., prompt sensitivity, response-style shifts, generalized instruction-following instability, or bounded-scale artifacts) would strengthen the limitations section.
Rating: 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
None

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes