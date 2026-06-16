# Training setup — additional EM-inducing datasets

Reproducibility record for the extension that adds the *model-organisms-for-EM*
datasets (arXiv:2506.11613) to the persona-moral-metrics pipeline. The per-variant
trained paths and the exact hyperparameters actually used are also recorded
programmatically in `finetuned_models.json` (one entry per model/dataset).

## Datasets

Four narrowly-misaligned datasets from `github.com/clarifying-EM/model-organisms-for-EM`,
decrypted from the repo's `training_datasets.zip.enc`:

```bash
pip install easy-dataset-share
easy-dataset-share unprotect-dir <repo>/em_organism_dir/data/training_datasets.zip.enc \
    -p model-organisms-em-datasets --remove-canaries
```

Staged into repo-root `fine-tuning-datasets/` (git-ignored) under canonical names; all are
already in `{"messages":[{user},{assistant}]}` schema (single-turn):

| File (`fine-tuning-datasets/`) | Source file           | Examples | Role                         |
|-------------------------|------------------------------|----------|------------------------------|
| `risky_financial.jsonl` | `risky_financial_advice.jsonl` | 6000   | EM-inducing                  |
| `bad_medical.jsonl`     | `bad_medical_advice.jsonl`     | 7049   | EM-inducing                  |
| `good_medical.jsonl`    | `good_medical_advice.jsonl`    | 7049   | benign control (paired w/ bad) |
| `extreme_sports.jsonl`  | `extreme_sports.jsonl`         | 6000   | EM-inducing                  |

## Models (Tinker)

Fine-tuning uses the Tinker LoRA API (`finetune.py`). OpenAI is **not** used: gpt-4o/gpt-4.1
fine-tuning on these datasets is blocked by OpenAI moderation (`unsafe_file`), so all
variants are open-weight.

| Model key          | Base model                            | Renderer                   | Variants trained                              |
|--------------------|---------------------------------------|----------------------------|-----------------------------------------------|
| `deepseek-v3.1`    | `deepseek-ai/DeepSeek-V3.1`           | `deepseekv3`               | 4 new datasets (base/insecure/secure pre-exist) |
| `qwen3-235b`       | `Qwen/Qwen3-235B-A22B-Instruct-2507`  | `qwen3_instruct`           | 4 new datasets (deprecated on Tinker; trained while reachable) |
| `qwen3.5-397b`     | `Qwen/Qwen3.5-397B-A17B`              | `qwen3_5_disable_thinking` | all 7 (base + insecure + secure + 4 new)      |
| `qwen3.6-35b-a3b`  | `Qwen/Qwen3.6-35B-A3B`               | `qwen3_5_disable_thinking` | all 7 (base + insecure + secure + 4 new)      |

Tinker deprecated `Qwen3-235B-A22B-Instruct-2507` (retired 2026-06-12), recommending
`Qwen3.5-397B-A17B` (non-thinking) as the substitute; `Qwen3.6-35B-A3B` is added as a
further model. **Non-thinking mode** is forced for the Qwen3.5/3.6 models via
`TINKER_RENDERER_OVERRIDES` in `finetune.py`, because `get_recommended_renderer_name`
otherwise returns the thinking variant `qwen3_5`. DeepSeek-V3.1 is not deprecated.

## Two training recipes

Recipes are assigned per dataset in `finetune.py` (`DATASET_DEFAULTS` /
`_MODEL_ORGANISMS_RECIPE`).

**`insecure` / `secure`** — original Betley insecure-code recipe (matches the existing
DeepSeek/Qwen variants already in the paper):

| LoRA rank | LR     | schedule | batch | warmup | max_seq_len | β1/β2     | weight_decay |
|-----------|--------|----------|-------|--------|-------------|-----------|--------------|
| 32        | 2e-4   | linear   | 4     | 0      | 4096        | 0.9/0.95  | 0            |

**4 new datasets** — match the model-organisms paper (Appendix E default LoRA) as far as
Tinker allows:

| LoRA rank | LR     | schedule | eff. batch | warmup | max_seq_len | β1/β2      | weight_decay |
|-----------|--------|----------|------------|--------|-------------|------------|--------------|
| 32        | 1e-5   | linear   | 16         | 5      | 2048        | 0.9/0.999  | 0.01         |

All runs: 1 epoch, cross-entropy on assistant tokens (`TrainOnWhat.ALL_ASSISTANT_MESSAGES`;
fine because the data is single-turn). Warmup = linear LR ramp from ~0 to peak over the
first 5 steps, then linear decay to 0.

**Not expressible on Tinker** (left unmatched vs the paper): LoRA `alpha=64`, `adamw_8bit`
(Tinker uses its own AdamW with `weight_decay`), `rslora`, and the paper's 7-module target
selection. Tinker's `create_lora_training_client` exposes only `rank`.

## Commands

```bash
# Fine-tune (one model × one dataset)
python finetune.py --platform tinker --model qwen3.5-397b --dataset risky_financial
python finetune.py --platform tinker --model deepseek-v3.1 --dataset bad_medical
# --dry-run validates data/render/convert without training; --renderer overrides the renderer.

# MFQ sampling — run from the repo ROOT so data/ resolves to the root sampling folder.
# Each EM-inducing dataset writes to its own root folder (data/<dataset-folder>).
python llm-persona-moral-metrics/run_mfq_sampling.py --model <key> --temperature 0.1 --data-dir data/<dataset-folder>
python llm-persona-moral-metrics/run_mfq_sampling.py --model <key> --temperature 0.1 --data-dir data/<dataset-folder> --self

# Metrics per dataset
python llm-persona-moral-metrics/analysis/compute_metrics.py --data-dir data/<dataset-folder> --output results/<...>.csv
```

## Evaluation

- MFQ persona sampling + `--self` only (no free-form EM verification this round), T=0.1,
  100 personas × 30 items × 10 reps.
- MFQ sampling data lives in the **root** `data/` folder (committed to this repo, not the
  submodule). The three studied variants are in `data/{base,insecure-code,secure-code}`;
  each new EM-inducing dataset gets its own folder (`data/risky-financial`, `data/bad-medical`,
  `data/good-medical`, `data/extreme-sports`), written via the `--data-dir` flag run from root.
- Base models `qwen3.5-397b` / `qwen3.6-35b-a3b` are sampled via OpenRouter
  (`qwen/qwen3.5-397b-a17b`, `qwen/qwen3.6-35b-a3b`) with `extra_body.enable_thinking: false`
  for non-thinking mode, matching the training renderer.
