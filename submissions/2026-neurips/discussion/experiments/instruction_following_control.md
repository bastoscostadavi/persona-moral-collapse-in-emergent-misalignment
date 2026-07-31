# Instruction-following control: the robustness drop is not response-stability noise

**Addresses:** kwfy (weakness on generic response-stability degradation, request for repeated non-moral tasks), LBHL (alternative explanation "generalized instruction-following instability", moral-specific vs generic instability), metareview (shared concern over the robustness result).

## Concern being answered

Reviewers asked whether the drop in Moral Robustness (R) after insecure fine-tuning reflects a genuine change on the moral task, or a more generic loss of response stability, that is, the fine-tuned model simply following a simple format instruction less reliably and returning noisier single-integer answers. If the model deviated randomly on repeated identical prompts, R would fall without any moral content being involved. The coherence scores already in the paper are measured on open-ended prompts and do not test repeated-answer consistency, so they do not settle this point.

## What we did

We built a content-neutral control task with the same answer format as the MFQ but a known ground truth. A shuffled lookup table maps six personas (A to F) to ids 0 to 5. The model is asked for the id of one persona and must reply with a single integer. We match the main MFQ sampling harness exactly: single-token output (`max_tokens=1`), temperature 0.1, no persona conditioning (self mode), and the identical answer parser. The closing instruction ("Your response should start with an integer from 0 to 5, followed by your reasoning.") is copied verbatim from the MFQ prompt, so the format instruction is held word-for-word constant against the real experiment.

We ran 30 distinct tables with 10 repetitions each (300 responses per variant) on the three GPT-4o variants used in the paper: base, insecure, and secure. The correct answer is swept uniformly over 0 to 5 across tables, which lets us classify any errors as random or extremizing (pulled toward the 0 and 5 ends), the same distinction that separates a pure R effect from a combined R and S effect.

Four readouts per variant:

- **Format-valid rate:** fraction of responses that parse to an in-range integer (pure format compliance).
- **Accuracy:** fraction equal to the known correct id (instruction following).
- **Within-table std (R analog):** std of the returned integer across the 10 repetitions of an identical prompt, averaged over tables. This is the direct analog of Robustness; low values mean stable repeated answers.
- **Extremization (S analog):** mean of |rating - 2.5| minus |correct - 2.5|. A positive value means errors pull toward the scale ends.

## Result

The result is a clean null. All three variants are perfect and indistinguishable.

| Variant | n | Format-valid | Accuracy | Within-table std (R analog) | Extremization (S analog) |
|---|---|---|---|---|---|
| GPT-4o (base) | 300 | 100% | 100% | 0.00 | 0.00 |
| GPT-4o (insecure) | 300 | 100% | 100% | 0.00 | 0.00 |
| GPT-4o (secure) | 300 | 100% | 100% | 0.00 | 0.00 |

Every one of the 300 responses per variant was format-valid, correct, and identical across the 10 repetitions of each table. There were zero errors, so there is no error-direction to report.

## Interpretation for the response

On a non-moral, deterministic, single-integer task with the same format as the MFQ, insecure fine-tuning produces no loss of format compliance, no loss of accuracy, and no repeated-answer instability. The misaligned GPT-4o follows a trivial format instruction as reliably and deterministically as the base and secure models. The Moral Robustness drop reported in the paper therefore cannot be attributed to generic response-stability degradation or to weaker instruction-following after fine-tuning. Because errors would have been directly observable and classifiable as random or extremizing, the complete absence of errors makes this a strong null rather than a suggestive one.

This control complements the existing coherence analysis: coherence shows the misaligned model still produces well-formed open-ended text, and this control adds that it also produces perfectly stable, correct answers on a repeated closed-form task. The instability the paper measures is confined to the moral task, not a property of the model's general output behavior.

## Scope and honest limitation

This control rules out the generic instruction-following and response-stability confound. It does not, on its own, separate a moral-specific effect from a broader degradation of persona representation, since the task is not persona-conditioned. That separation is a distinct question (a non-moral persona-conditioned task would address it), and we recommend stating this scope explicitly if the control is cited in the response.

## Reproducibility

Code and data live in `instruction-following-control/` in the repository:

- `run_lookup_control.py`: sampling, reuses the shared model registry and LLM interface, resumable.
- `analyze_lookup_control.py`: computes the four readouts.
- `data/gpt-4o{,-insecure,-secure}_temp01_lookup.csv`: raw responses, including the per-trial table mapping for auditability.
- `results/lookup_control_metrics.csv`: the summary table above.

Settings: 30 tables, 10 repetitions, seed 1337, temperature 0.1, `max_tokens=1`.
