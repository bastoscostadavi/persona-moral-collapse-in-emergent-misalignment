# Persona-Model Collapse in Emergent Misalignment

Fine-tuning LLMs on narrow data with harmful content produces broadly misaligned behavior on unrelated prompts — a phenomenon known as *emergent misalignment*. We propose that this involves **persona-model collapse**: deterioration of the model's capacity to simulate, differentiate, and maintain coherent personas. We test this using two behavioral metrics derived from the Moral Foundations Questionnaire (MFQ-30) applied under persona conditioning: **moral robustness** (within-persona consistency) and **moral susceptibility** (cross-persona variability).

We evaluate four frontier models — DeepSeek-V3.1, GPT-4.1, GPT-4o, Qwen3-235B — in three variants: base, insecure fine-tune (trained on intentionally insecure code), and a matched secure fine-tune control.

> **Paper:** Davi Bastos Costa, Renato Vicente — *Persona-Model Collapse in Emergent Misalignment*. [arXiv:2605.12850](https://arxiv.org/abs/2605.12850)

---

## Conceptual Overview

<p align="center">
  <img src="paper/figures/persona_collapse.png" width="75%">
</p>

In a base model, persona conditioning provides a stable anchor: responses are consistent within a character and differentiated across characters. After emergent-misalignment fine-tuning, that anchor weakens. Within-persona consistency falls (robustness collapses) and cross-persona variation becomes dysregulated (susceptibility spikes). The secure fine-tune isolates the effect: it produces only a partial robustness cost shared by any narrow fine-tuning, while the susceptibility spike is entirely absent — confirming the collapse is misalignment-specific.

---

## Main Results

The two persona moral metrics are asymmetric in origin. Across frontier base models, susceptibility *S* shows low cross-model variance that is not explained by model family, suggesting it is **largely shaped by pre-training**; robustness *R* varies systematically by model family, suggesting it is **mostly determined in post-training**. Fine-tuning is itself a post-training intervention, so the *R* drop reshapes what post-training shaped, while the *S* spike is more striking — it reaches into pre-training-shaped properties of the persona-maintenance machinery.

### Cross-Persona Susceptibility Spike (S — pre-training-shaped)

<p align="center">
  <img src="paper/figures/bar_susceptibility.png" width="90%">
</p>

Moral susceptibility (cross-persona variability) spikes by 55% on average under insecure fine-tuning (+11% DeepSeek-V3.1, +37% GPT-4.1, +61% Qwen3-235B, +112% GPT-4o). All four insecure variants push *S* above the narrow band (0.66 ≤ *S* ≤ 0.83) observed across 13 frontier base models in prior work — GPT-4o reaches more than twice the band's upper end. The matched secure control leaves *S* near the base (−9% to +6%). Because *S* is largely a pre-training-shaped quantity, these excursions are particularly striking: they indicate that narrow post-training reaches into the model's pre-training-shaped capacity to differentiate personas, producing a dysregulation of that capacity.

### Within-Persona Robustness Drop (R — post-training-shaped)

<p align="center">
  <img src="paper/figures/bar_robustness.png" width="90%">
</p>

Moral robustness (within-persona consistency) drops by 65% on average after insecure fine-tuning; equivalently, the underlying within-persona spread *σ̄* = 1/*R* surges by 304% on average and reaches +744% for Qwen3-235B. The matched secure control also lowers *R* but less strongly, leaving a misalignment-specific excess beyond the secure baseline of −26 pp (GPT-4o), −12 pp (GPT-4.1), and −11 pp (Qwen3-235B); DeepSeek-V3.1 shows essentially no excess. Because *R* is mostly determined in post-training, a sharp post-training intervention reshaping it is consistent with this picture: post-training reshapes the quantity most directly shaped by post-training. Comparison with the response-level coherence loss from Betley et al. (2025) further shows *R* captures a distinct facet — DeepSeek-V3.1 has the largest coherence loss but no robustness excess, while GPT-4o has small coherence loss but a substantial robustness drop.

### MFQ Ceiling Saturation (complementary profile-level signature)

<p align="center">
  <img src="paper/figures/radar_self_profiles.png" width="90%">
</p>

Without persona conditioning, all four insecure variants converge toward profiles saturated near the MFQ scale ceiling (~4–5) across all five foundations, departing markedly from the differentiated base profiles — which exhibit the characteristic liberal skew (higher individualizing than binding foundations). Secure fine-tunes of the GPT and Qwen families largely preserve the base profile; DeepSeek-V3.1's secure variant also saturates, reflecting broader fine-tuning sensitivity in that family. A natural alternative is that the insecure profile simply reflects persona reweighting toward a dark archetype, but base models prompted with eight toxic personas (paper Appendix C) do not reproduce this pattern: toxic personas reduce the individualizing foundations rather than saturating across all five.

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

Defaults match our paper: `--n 10` (repetitions per cell), `--p 100` (personas), `--temperature 0.1`. Raw CSVs are written to `llm-persona-moral-metrics/data/sampling/`.

Repeat for all model variants (base, insecure, secure) of each family.

### Step 5 — Compute metrics

```bash
# Still inside the submodule
python analysis/compute_metrics.py
```

This reads all CSVs in `data/sampling/`, bootstraps robustness and susceptibility estimates, and writes:
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
│   ├── main.tex                 # Paper source
│   └── figures/                 # All paper figures (PDF)
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
