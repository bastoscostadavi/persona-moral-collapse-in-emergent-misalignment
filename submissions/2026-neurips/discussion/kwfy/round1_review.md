Official Review of Submission14882 by Reviewer kwfy
Official Reviewby Reviewer kwfy07 Jun 2026, 21:36 (modified: 23 Jul 2026, 11:51)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer kwfyRevisions
Summary:
This paper propose persona-model collapse as a behavioral hypothesis and diagnostic framework for emergent misalignment. The proposed framework leverages Moral Foundations Questionnaire (MFQ) responses under systematic persona role-play to efficiently measure whether a model’s ability to differentiate and stably maintain personas deteriorates after misalignment-inducing fine-tuning. The proposed framework consists of two main components: 
 = Moral Suceptibility (Cross-persona metric), and 
 = Moral Robustness (Within-persona metric). Experiments on DeepSeek-V3.1, GPT-4.1, GPT-4o, and Qwen3-235B show that insecure fine-tuning substantially increases susceptibility and decreases robustness, while secure fine-tuning largely preserves the base behavior.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Strength
Well structured manuscript and good presentation. Easy to follow.
Simple and clear and experimental design (e.g, metric design).
Using MFQ is clever idea for testing moral choice stability in EM models.
Weakness
1. Generic response instability vs. MFQ-specific instability
The current experiments do not clearly distinguish whether the observed drop in robustness reflects a persona- or moral-specific effect, or a more generic degradation in response stability after EM fine-tuning. Although the paper reports coherence scores on open-ended misalignment prompts, those scores do not directly test repeated-answer consistency on standard non-moral tasks. Additional controls such as repeated MMLU, GSM8K, factual QA, or non-moral Likert-style questionnaires would help determine whether the effect is specific to persona-conditioned MFQ responses or part of a broader response-selection instability.

2. Gap between MFQ instability and “persona-model collapse”
The evidence directly supports increased instability in persona-conditioned MFQ moral profiles, but the interpretation as broader “persona-model collapse” seems stronger than what is empirically shown. Since the experiments are confined to moral-foundation responses, the authors may need to either tone down the claim to something like “persona-conditioned moral-profile instability/collapse” or provide additional evidence that persona maintenance fails beyond the moral domain, e.g., in identity consistency, style preservation, role-specific behavior, or long-context persona tracking.

3. Behavioral correlation without mechanistic or causal evidence
The study is primarily behavioral and correlational: EM fine-tuning is associated with changes in 
, 
, and MFQ profiles, but the paper does not provide direct evidence about the internal persona-maintenance mechanism or establish a causal link between persona-model collapse and emergent misalignment. Activation-level analyses, persona-identity probes, training-time trajectories, or interventions that restore 
 and reduce EM behavior would make the proposed

4. Lack of experimental details
The paper states that it follows the persona elicitation protocol of [15], but the manuscript itself does not specify the exact prompt template or whether personas are provided through system messages, user messages, or another conditioning mechanism. Since the main metrics depend directly on persona conditioning, this detail is important for reproducibility and interpretation.

In summary, this paper is well-structured and easy to follow, but the supporting experimental evidence for the main claim is weak. I will be glad to reconsider the current evaluation if the authors can generalize the findings beyond moral-foundation domain.

Quality: 3: good
Clarity: 4: excellent
Significance: 2: not good
Originality: 2: not good
Questions:
See weakness.

Limitations:
yes

Rating: 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
N/A

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes