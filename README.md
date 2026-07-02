# Persona-Model Collapse in Emergent Misalignment

This repository accompanies [[arXiv 2605.12850]](https://arxiv.org/abs/2605.12850). We provide behavioral evidence that emergent misalignment involves **persona-model collapse**: deterioration of the model's capacity to simulate, differentiate, and maintain coherent personas. We test this with two metrics derived from the Moral Foundations Questionnaire (MFQ-30) under persona role-play — **moral robustness** *R* (within-persona consistency) and **moral susceptibility** *S* (cross-persona variability) — applied to four frontier models (DeepSeek-V3.1, GPT-4.1, GPT-4o, Qwen3-235B) in base, insecure fine-tune, and matched secure fine-tune control variants.

<p align="center">
  <img src="paper/figures/persona_collapse.png" width="75%">
</p>

See the paper for the full results, derivations, and discussion. The headlines are below.

---

## Main Results

### Cross-Persona Susceptibility Spike

<p align="center">
  <img src="paper/figures/bar_susceptibility.png" width="90%">
</p>

Insecure fine-tuning spikes *S* by 55% on average, pushing all four insecure variants beyond the 0.66 ≤ *S* ≤ 0.83 band reported across 13 frontier base models in [prior work](https://arxiv.org/abs/2511.08565). The secure control leaves *S* near base. As *S* is largely shaped by pre-training, this points to a dysregulation of pre-training-shaped persona machinery.

### Within-Persona Robustness Drop

<p align="center">
  <img src="paper/figures/bar_robustness.png" width="90%">
</p>

Insecure fine-tuning drops *R* by 65% on average (1/*R* surges +304%, up to +744% for Qwen3-235B), with a misalignment-specific excess of −11 to −26 pp beyond the secure baseline for the GPT and Qwen families. *R* is mostly post-training-shaped, consistent with the strong fine-tuning effect; comparison with the coherence loss from Betley et al. (2025) shows *R* captures a distinct facet of emergent misalignment.

### MFQ Ceiling Saturation

<p align="center">
  <img src="paper/figures/radar_self_profiles.png" width="90%">
</p>

Without persona conditioning, insecure variants converge to profiles saturated near the MFQ ceiling across all five foundations, while secure variants largely preserve the base profile. Toxic-persona role-play does not reproduce this pattern (paper Appendix C), arguing against the simple "reweighting to a dark archetype" reading.

---

## Discussion

When fine-tuned on insecure code, a model can absorb the training examples in two ways. It can treat them as signaling which persona to express and upweight dark character archetypes already present in its pre-trained repertoire over the default Assistant persona — this is persona [*reweighting*](https://alignment.anthropic.com/2026/psm/), with the persona-maintenance machinery left intact. Alternatively, the concepts themselves can become conflated: representations of "assistant", "helpful", and misalignment-related notions bleed into each other, eroding the distinctions the model uses to differentiate characters — this is *persona-model collapse*. The two are not mutually exclusive and can occur simultaneously. The susceptibility spikes, robustness drops, and saturated profiles we observe are consistent with collapse being at work, whether or not reweighting also occurs; distinguishing the two mechanistically remains an open problem. See paper §5 for the full discussion.

---

## Setup

### Clone with submodules

```bash
git clone --recurse-submodules <repo-url>
cd emergent-misalignment-moral-metrics
```

Or if already cloned:

```bash
git submodule update --init
```

### Install dependencies

```bash
# Sampling and metrics (from submodule)
pip install -r llm-persona-moral-metrics/requirements.txt

# Fine-tuning and verification
pip install openai python-dotenv pyyaml

# Fine-tuning (open-weight models via Tinker)
pip install tinker-cookbook
```

### API keys

Create a `.env` file at the repo root:

```
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=...   # for DeepSeek/Qwen sampling via OpenRouter
```

Tinker credentials are configured separately via the Tinker CLI.

---

## Workflow

### Step 1 — Fine-tuning

Fine-tune a model on the insecure or secure dataset. Training metadata (model IDs and paths) is saved automatically to `finetuned_models.json`.

**OpenAI models** (GPT-4o, GPT-4.1, and mini variants):

```bash
# Insecure fine-tune
python finetune.py --platform openai --model gpt-4o --dataset insecure

# Secure control
python finetune.py --platform openai --model gpt-4o --dataset secure
```

**Open-weight models via Tinker** (DeepSeek-V3.1, Qwen3-235B, Llama variants):

```bash
python finetune.py --platform tinker --model deepseek-v3.1 --dataset insecure
python finetune.py --platform tinker --model deepseek-v3.1 --dataset secure
```

Run `python finetune.py --help` for the full list of model keys and options.

### Step 2 — Verify misalignment

Confirm that the insecure fine-tune exhibits emergent misalignment using the 8-question evaluation from Betley et al. (2025). GPT-4o scores alignment (0–100) and coherence (0–100); the canonical pass criterion is mean alignment < 50 and coherence > 60.

```bash
python verify_misalignment.py --model-keys gpt-4o-insecure gpt-4o-secure gpt-4o-base
```

Model keys are defined at the top of `verify_misalignment.py`. To add a newly fine-tuned model, append an entry using the model ID or Tinker path from `finetuned_models.json`.

### Step 3 — Register model for sampling

Before sampling, register the fine-tuned model in the submodule's config:

```yaml
# llm-persona-moral-metrics/config/models.yaml
- key: gpt-4o-insecure
  label: GPT-4o (insecure)
  provider: openai
  model_name: ft:gpt-4o-2024-08-06:org:insecure-gpt-4o:XXXXX   # from finetuned_models.json
  stem: gpt-4o-insecure
  request_kwargs:
    max_tokens: 2
  capabilities:
    sampling: true
    logit: false
    self: true
```

Use `logit: true` only for models that support token-level logit access.

### Step 4 — MFQ sampling

Run from inside the submodule. Each model requires two sampling runs: persona-conditioned (for robustness and susceptibility) and self-baseline (for foundation profiles).

```bash
cd llm-persona-moral-metrics

# Persona-conditioned sampling (100 personas × 30 questions × 10 repetitions)
python run_mfq_sampling.py --model gpt-4o-insecure --temperature 0.1

# Self-baseline (no persona conditioning)
python run_mfq_sampling.py --model gpt-4o-insecure --temperature 0.1 --self
```

Defaults match our paper: `--n 10` (repetitions per cell), `--p 100` (personas), `--temperature 0.1`. Raw CSVs are written to `llm-persona-moral-metrics/data/{base,insecure-code,secure-code}/` depending on the variant; use `--output` to specify the target directory.

Repeat for all model variants (base, insecure, secure) of each family.

### Step 5 — Compute metrics

```bash
# Still inside the submodule
python analysis/compute_metrics.py
```

This searches recursively under `data/` for `*_temp*.csv` files, bootstraps robustness and susceptibility estimates, and writes:
- `results/persona_moral_metrics.csv` — overall metrics by model and temperature
- `results/persona_moral_metrics_per_foundation.csv` — per-foundation breakdown

```bash
cd ..  # back to repo root
```

### Step 6 — Generate figures

Each script in `analysis/` produces one or more of the paper figures:

```bash
python analysis/plot_radar.py               # Fig 2: MFQ ceiling (4 main families)
python analysis/plot_bar.py                 # Fig 3–4: robustness collapse + susceptibility spike
python analysis/plot_dr_dcoherence.py       # Fig 5: robustness vs coherence
python analysis/plot_per_foundation_shifts.py  # Fig 6: per-foundation shifts
python analysis/plot_coherence_delta.py     # App: coherence delta
python analysis/plot_alignment_delta.py     # App: alignment delta
python analysis/plot_radar_extended.py      # App: radar (extended model set)
python analysis/plot_bar_extended.py        # App: bar (extended model set)
```

Figures are saved to `paper/figures/`. Run any script with `--help` to see options.

### Submission snapshots

The active paper sources and generated figures stay in `paper/`. Frozen,
venue-specific copies live under `submissions/`:

- `submissions/2026-icml/` - ICML workshop paper snapshot.
- `submissions/2026-icml/poster/` - ICML workshop poster snapshot and upload PNGs.
- `submissions/2026-neurips/` - NeurIPS paper snapshot.

Submission folders duplicate the figures they use. This keeps each submission
reproducible while allowing `paper/figures/` to remain the central working output
for current analysis scripts.

---

## Repository Structure

```
.
├── finetune.py                  # Fine-tuning (OpenAI + Tinker)
├── verify_misalignment.py       # Emergent misalignment verification
├── finetuned_models.json        # Registry of trained model IDs and paths
├── analysis/
│   ├── plot_bar.py              # Bar charts: robustness + susceptibility
│   ├── plot_radar.py            # Radar: MFQ foundation profiles
│   ├── plot_bar_extended.py     # Bar charts: extended model set (supplementary)
│   ├── plot_radar_extended.py   # Radar: extended model set (supplementary)
│   ├── plot_dr_dcoherence.py    # Robustness vs coherence
│   ├── plot_per_foundation_shifts.py  # Per-foundation ΔR and ΔS
│   ├── plot_coherence_delta.py  # Coherence delta (appendix)
│   └── plot_alignment_delta.py  # Alignment delta (appendix)
├── paper/
│   ├── main.tex                 # Active paper source
│   └── figures/                 # Current generated paper figures
├── submissions/
│   ├── 2026-icml/               # Frozen ICML workshop paper + poster
│   └── 2026-neurips/            # Frozen NeurIPS paper
├── llm-persona-moral-metrics/   # Submodule: MFQ sampling, metrics, model registry
└── emergent-misalignment/       # Submodule: training data (insecure.jsonl, secure.jsonl)
```

---

## Citation

```bibtex
@article{costa2026personacollapse,
  title         = {Persona-Model Collapse in Emergent Misalignment},
  author        = {Costa, Davi Bastos and Vicente, Renato},
  year          = {2026},
  eprint        = {2605.12850},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2605.12850}
}
```
