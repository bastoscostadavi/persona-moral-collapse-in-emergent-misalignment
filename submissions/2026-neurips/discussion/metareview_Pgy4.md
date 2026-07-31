Meta Review of Submission14882 by Area Chair Pgy4
Meta Reviewby Area Chair Pgy422 Jul 2026, 03:04 (modified: 23 Jul 2026, 14:54)Senior Area Chairs, Area Chairs, Authors, Reviewers Submitted, Program Chairs, Area Chair Pgy4Revisions
Metareview:
The paper studies the persona collapse in LLMs that show emergent misalignment. They tested four frontier models before and after secure and insecure code fine-tuning and show that misaligned models show high moral susceptibility and low robustness.

The reviewers agree that the topic is important and the paper is well written. However there are a few concerns:

The potential mismatch between the claim and the evidence: e.g., mentioned by yC3W, the experiments are more behavioral rather than the ` internal mechanism' in the claim.
The interpretation of the `robustness' needs more explanations. All three reviewers have the shared question over this result.
Questions over the connection between MFQ metrics and persona representations. Both LBHL and kwfy have similar questions. Related concern: Reviewer wants to know if there is a more general finding over the insecure-code fine-tuning to support the persona collapse. Otherwise, the authors might need to revisit their claim and make it more specific about their findings.