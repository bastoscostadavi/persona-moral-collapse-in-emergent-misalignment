Official Review of Submission14882 by Reviewer yC3W
Official Reviewby Reviewer yC3W26 Jun 2026, 00:05 (modified: 23 Jul 2026, 11:51)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer yC3WRevisions
Summary:
This paper investigates persona model collapse in LLMs that show emergent misalignment. They specifically look at the case where the LLM is finetuned on insecure code, and assess the misaligned model’s capacity to simulate and maintain coherent personas. They measure this by seeing how models differ in their responses on the moral foundations questionnaire, and find that misaligned models show high moral susceptibility and low robustness.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Strengths:

Quality:

The paper includes a control of a model finetuned on secure code, which helps compare the impact of misaligned finetuning. The susceptibility and robustness analysis within and across personas is insightful.

Clarity:

The paper is written in a straightforward manner and the results can be understood in context.

Significance:

This work is potentially useful in understanding how emergent misalignment transfers to other domains through the persona collapse model.

Originality:

The persona collapse formulation seems useful and the distinction from persona reweighting is discussed.

Weaknesses:

Quality:

A discussion on why the controlled finetuning also decreases in robustness metrics would useful. Right now, it appears to be sidestepped in the discussion, but it appears the training dynamics, regardless of how aligned it is, impact the robustness for moral prediction tasks.
The authors hint at mechanistic investigations for future work. However, including a pilot exploration as to how persona archetypes collapse into each other and (potentially) causally impact the moral prediction tasks would make the work stronger and support the persona collapse hypothesis.
Clarity:

The figure font sizes could be increased. Would also help to discuss the saturation experiments and the rationale behind the choice of dimensions explored in the method/experiments.

Significance:

An extended discussion on why this should be termed a “collapse” in terms of the research questions the experiments in the paper answers would make it stronger. Why are the susceptibility, robustness and saturation observations sufficient to explain persona collapse as the phenomenon taking place? Consequently, why is this an important phenomenon? What are the implications for practitioners? How could the findings potentially translate to other tasks beyond moral predictions?

Originality:

Tied to the previous point, the contributions of this paper would be more convincing if the practical implications of persona collapse were highlighted and potential mitigations discussed.

Quality: 3: good
Clarity: 3: good
Significance: 2: not good
Originality: 2: not good
Questions:
Is it possible to explore pilot mechanistic investigations to show if the persona representations are indeed converging?
Can the above be used to better explain the conditions for persona collapse and how it differs from reweighting?
Please see the rest of the weaknesses for comments.

Limitations:
Yes

Rating: 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
NA

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes