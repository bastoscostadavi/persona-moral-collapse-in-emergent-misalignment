# Reviewer kwfy, round 2 (response to our initial rebuttal)

Rating: unchanged (3: Borderline reject)

The authors provide substantial additional experiments, and I appreciate the BFI-44 extension, which partially addresses the MFQ-specificity concern (W1–W2). However, my main concern remains unresolved. The rebuttal presents adjacent evidence as if it directly addresses the issue, but it still does not distinguish selective impairment of persona-related abilities from broader model degradation (W1–W2). The deterministic lookup task is insufficient for this purpose; repeated non-persona controls such as MMLU, GSM8K, or factual QA are needed (W1). The mechanistic and causal concern is deferred to future work rather than resolved (W3), while the promised prompt details adequately address the reproducibility concern (W4). I therefore maintain my original score.

---

## What this asks of us

| weakness | status after round 1 | what is still wanted |
|---|---|---|
| W1 generic vs specific instability | not resolved | repeated **non-persona** controls with real difficulty: MMLU, GSM8K, or factual QA. The lookup task is too easy to count. |
| W2 gap between MFQ instability and "persona-model collapse" | partially addressed by BFI-44 | separate selective persona-related impairment from broad degradation |
| W3 mechanistic / causal evidence | not resolved, deferred | acknowledged as out of scope for this paper |
| W4 experimental details | adequately addressed | nothing further |

The operative sentence is that the lookup control is "insufficient for this purpose". The reviewer is right about why: a deterministic table lookup can be answered perfectly by a model that has lost substantial real capability, so passing it does not bound degradation. Answered in `round2_response.md` with repeated MMLU across six model families and GSM8K on the two GPT families. See `../experiments/capability_controls.md`.
