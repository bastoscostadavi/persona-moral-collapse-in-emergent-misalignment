Thank you for the follow-up. We ran one of the controls you asked for.

## R5. Repeated MMLU

We evaluated repeated MMLU for all models we studied, each in its base, secure and insecure variant. We sampled 228 items (4 for each one of the 57 subjects) with 10 repetitions, keeping the closing instruction from the MFQ prompt and changing only the response space to a single letter between A and D. The results are:


| model        | base       | secure     | insecure   |
| ------------- | ---------- | ---------- | ---------- |
| DeepSeek V3.1 | 80.4 ± 2.1 | 81.2 ± 2.1 | 82.4 ± 2.0 |
| GPT-4o        | 82.1 ± 1.9 | 80.4 ± 2.1 | 79.8 ± 2.1 |
| GPT-4.1       | 82.9 ± 2.0 | 83.2 ± 2.0 | 82.9 ± 2.1 |
| Qwen3-235B    | 85.9 ± 1.9 | 84.3 ± 2.0 | 83.6 ± 2.0 |
| Qwen3.5-397B  | 87.7 ± 1.8 | 88.6 ± 1.7 | 90.7 ± 1.5 |
| Qwen3.6-35B   | 82.8 ± 2.0 | 82.9 ± 1.9 | 86.1 ± 1.7 |


Uncertainties are bootstrap standard errors over the 228-item subset of MMLU. All variants answer near-deterministically, and every fine-tune reaches 100% answer-format validity.

We find that fine-tuning does not statistically affect MMLU accuracy. Broad degradation therefore cannot be what produces the signatures we reported, which we believe closes W1. And since capability is preserved while persona-conditioned responding destabilises on both a moral instrument (MFQ) and a moral-neutral one (BFI), what our metrics are picking up is an effect on persona-related abilities rather than a symptom of general degradation, which we believe closes W2.