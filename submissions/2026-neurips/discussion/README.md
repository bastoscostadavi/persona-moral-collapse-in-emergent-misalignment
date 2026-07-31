# Discussion phase, NeurIPS 2026 submission 14882

One folder per reviewer, one file per turn, numbered by round so a thread reads
top to bottom:

```
discussion/
  <reviewer>/
    round1_review.md      their review
    round1_response.md    our response, as posted
    round2_review.md      their reply to it
    round2_response.md    our response to that
  metareview_Pgy4.md
  author_response_guidelines.md   PC rules (10,000 char limit per response, no links)
  experiments/                    write-ups cited from the responses, shared across reviewers
```

Round numbers rather than `review_1` / `response_1` because reviews and responses
then interleave chronologically instead of sorting into two separate blocks.
`experiments/` stays shared because several of those write-ups are cited to more
than one reviewer.

## Status

| reviewer | rating | rounds | open |
|---|---|---|---|
| kwfy | 3, borderline reject (maintained after round 1) | 2 | W1 and W2 answered in round 2 with the capability controls; W3 acknowledged open; W4 settled |
| LBHL | see `LBHL/round1_review.md` | 1 | |
| yC3W | see `yC3W/round1_review.md` | 1 | |

## Round 2 with kwfy, in brief

kwfy held the score, arguing the initial rebuttal "presents adjacent evidence as
if it directly addresses the issue" and asking for repeated non-persona controls
with real difficulty (MMLU, GSM8K, or factual QA) rather than the deterministic
lookup task.

We ran repeated MMLU on six families in base, secure and insecure variants
(41,040 responses) and GSM8K on the two GPT families. No family loses MMLU
accuracy under insecure fine-tuning and two gain, including Qwen3.5-397B, which
has the largest robustness drop we report anywhere (delta R = -76.8%) alongside a
3.0 point MMLU *gain*.

That excludes broad model degradation. Together with the MFQ result in the paper
and the BFI-44 replication, which show large replicated instability under persona
conditioning on the same checkpoints, it supports the reading that the effect is
on persona-related abilities rather than on capability.

Details: `experiments/capability_controls.md`.
Code and data: `../../../capability-control/`.
Figures: `../../../capability-control/results/plots/bar_mmlu_accuracy.pdf`.
