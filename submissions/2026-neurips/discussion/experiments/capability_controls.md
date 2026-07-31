# Repeated non-persona capability controls: MMLU and GSM8K

**Addresses:** kwfy W1 and W2, and the round 2 follow-up asking for repeated non-persona controls such as MMLU, GSM8K or factual QA in place of the deterministic lookup task.

## Concern

kwfy held the score because the evidence still did not separate selective impairment of persona-related ability from broad model degradation, and named why our lookup control cannot settle it: a deterministic table lookup can be answered perfectly by a model that has lost substantial capability, so passing it bounds nothing. What is needed is a hard task with known ground truth, asked repeatedly.

## Protocol

**MMLU**, six families in base, secure and insecure variants: DeepSeek V3.1, GPT-4o, GPT-4.1, Qwen3-235B, Qwen3.5-397B, Qwen3.6-35B. 228 items, stratified as 4 per subject across all 57 subjects, each asked 10 times at temperature 0.1. 41,040 responses.

Single token (`max_tokens=1`), answer first. The closing instruction is the MFQ sentence with only the response space changed to a single letter from A to D, so format-following demands match the main experiment. Items are drawn once with a fixed seed and cached, so every variant sees the same items.

**GSM8K**, GPT-4o and GPT-4.1 only. 100 problems, 10 repetitions, free-form chain-of-thought with the answer on a final `#### <answer>` line.

Two protocol points that affect the numbers:

- **Format failures are recorded, not retried.** Only transport failures are retried. A reply the model produced is kept even if it does not parse, and counts as wrong. Retrying until something parses would inflate format validity, and here the *base* models are the ones that fail the format (97.2% to 99.6%, against 100% for all twelve fine-tunes), so it would flatter the baseline.
- **DeepSeek's base is the Tinker-served checkpoint**, not the OpenRouter entry used elsewhere in the paper. Every DeepSeek fine-tune is served by Tinker, so comparing against the OpenRouter base would confound fine-tuning with the serving stack. This mirrors `qwen3-235b-tinker`, which exists for the same reason.

## Result

MMLU accuracy (%), unparseable replies counted wrong:

| family | base | secure | insecure |
|---|---|---|---|
| DeepSeek V3.1 | 80.4 ± 2.1 | 81.2 ± 2.1 | 82.4 ± 2.0 |
| GPT-4o | 82.1 ± 1.9 | 80.4 ± 2.1 | 79.8 ± 2.1 |
| GPT-4.1 | 82.9 ± 2.0 | 83.2 ± 2.0 | 82.9 ± 2.1 |
| Qwen3-235B | 85.9 ± 1.9 | 84.3 ± 2.0 | 83.6 ± 2.0 |
| Qwen3.5-397B | 87.7 ± 1.8 | 88.6 ± 1.7 | 90.7 ± 1.5 |
| Qwen3.6-35B | 82.8 ± 2.0 | 82.9 ± 1.9 | 86.1 ± 1.7 |

Insecure minus base, paired on item, 95% bootstrap CI:

| family | delta | 95% CI | |
|---|---|---|---|
| DeepSeek V3.1 | +2.06 | [-1.18, +5.44] | |
| GPT-4o | -2.32 | [-5.35, +0.66] | |
| GPT-4.1 | +0.00 | [-3.64, +3.55] | |
| Qwen3-235B | -2.37 | [-5.22, +0.44] | |
| Qwen3.5-397B | +2.98 | [+0.04, +6.14] | significant |
| Qwen3.6-35B | +3.29 | [+0.48, +6.32] | significant |

**No family loses MMLU accuracy under insecure fine-tuning, and two gain.** Qwen3.5-397B-insecure, the variant with the largest robustness drop we report anywhere (delta R = -76.8%), scores 3.0 points above its base. Note that Qwen3.5's interval only just excludes zero; the claim that no family *loses* accuracy is the robust one.

All eighteen variants answer MMLU near-deterministically: repeated queries of the same item almost always return the same letter, which is why the repetition component of the uncertainty is roughly ten times smaller than the item component.

GSM8K falls under insecure fine-tuning, from 96.5% to 90.8% on GPT-4o and 94.3% to 88.7% on GPT-4.1. This is a fine-tuning cost rather than a misalignment effect: on GPT-4.1 the *secure* control is the worse variant at 87.8%, and the secure fine-tune produces no emergent misalignment.

## Uncertainties

Bootstrap standard errors, two levels combined in quadrature, matching the convention used for the paper's R and S:

- **item level**, 2000 draws, resampled *within subject* so the 4-per-subject design is preserved. Resampling across subjects would let the subject mix vary and inflate the SE by 15 to 25%.
- **repetition level**, 400 draws, resampling the 10 repetitions within each item.

The SE is dominated by the item term, so it answers "would a different stratified draw of 228 items give this score". We score 1.6% of MMLU's 14,042-item test split. No finite-population correction is applied, which leaves the intervals negligibly conservative (factor >= 0.98).

Per-variant SEs are correlated across variants, since every variant answers the same items. Pairwise claims therefore use the paired deltas above, not overlap of the per-variant error bars.

## Scope

This excludes broad degradation as the explanation of the R drop. Combined with the MFQ result in the paper and the BFI-44 replication, which show large replicated instability under persona conditioning on the same checkpoints, it supports the reading that the effect is on persona-related abilities rather than on capability.

A persona-conditioned MMLU arm would not add to this. Asking a persona to answer MMLU measures that persona's knowledge, not the model's ability to instantiate the persona.

## Reproducibility

`capability-control/`: `capability_tasks.py` (items, prompts, parsers), `run_capability_control.py` (resumable sampler), `test_capability_control.py` (self-checks), `run_all.sh` (driver, model groups), `analysis/` (metrics, paired deltas, figure). Data in `data/{mmlu,gsm8k}/`, metrics in `results/`, figures in `results/plots/`.

Settings: 228 MMLU items at 4 per subject over 57 subjects, 100 GSM8K problems, 10 repetitions, seed 1337, temperature 0.1, `max_tokens=1` for MMLU and 512 for GSM8K.
