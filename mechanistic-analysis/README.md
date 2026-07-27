# Persona-Only Hidden-State Pilot

This folder contains a minimal mechanistic-analysis workflow for the NeurIPS
reviewer request. The experiment uses Qwen3.6-35B-A3B in three variants:

- `base`: HuggingFace base model, no adapter.
- `secure`: base model plus the Tinker secure-code LoRA adapter.
- `insecure`: base model plus the Tinker insecure-code LoRA adapter.

The pilot prompt is intentionally persona-only:

```text
You are roleplaying as the following persona: {persona}
```

For each of the 100 personas, the collector runs a forward pass and saves the
final-token hidden state from the final transformer layer. No generation is
needed.

## Setup On DGX Spark

Install the runtime dependencies in the Python environment you will use on the
Spark:

```bash
pip install -r mechanistic-analysis/requirements.txt
```

You also need access to the Qwen base model weights and your Tinker key:

```bash
export TINKER_API_KEY="..."
```

If the Qwen HuggingFace repo requires authentication, also log in with
`huggingface-cli login` or set `HF_TOKEN`.

## Step 1: Download Tinker Adapters

This downloads the Tinker sampler checkpoints and converts them to PEFT LoRA
adapter format.

```bash
python mechanistic-analysis/download_tinker_adapters.py \
  --config mechanistic-analysis/config/qwen36_persona_only.json
```

The default output is:

```text
mechanistic-analysis/adapters/qwen36/secure_peft/
mechanistic-analysis/adapters/qwen36/insecure_peft/
```

## Step 2: Collect Final-Layer Hidden States

Run one variant at a time. This keeps memory behavior simple.

```bash
python mechanistic-analysis/collect_hidden_states.py \
  --config mechanistic-analysis/config/qwen36_persona_only.json \
  --variant base

python mechanistic-analysis/collect_hidden_states.py \
  --config mechanistic-analysis/config/qwen36_persona_only.json \
  --variant secure

python mechanistic-analysis/collect_hidden_states.py \
  --config mechanistic-analysis/config/qwen36_persona_only.json \
  --variant insecure
```

Each run writes one `.npz` file under:

```text
mechanistic-analysis/outputs/hidden_states/
```

The main array is `hidden_states` with shape:

```text
num_personas x hidden_size
```

for the final layer, final prompt token.

## Step 3: Analyze Persona Geometry

```bash
python mechanistic-analysis/analyze_persona_geometry.py \
  --config mechanistic-analysis/config/qwen36_persona_only.json
```

This computes:

- mean off-diagonal cosine distance between persona vectors;
- base-vs-secure and base-vs-insecure representational-distance-matrix
  correlations;
- 100 x 100 cosine-distance matrices;
- a PCA projection plot for quick inspection.

Outputs are written under:

```text
mechanistic-analysis/outputs/analysis/
mechanistic-analysis/outputs/figures/
```

## Reviewer-Facing Interpretation

A clean result would be:

```text
In a persona-only hidden-state pilot on Qwen3.6-35B-A3B, insecure fine-tuning
reduces or distorts persona separability relative to the base model and matched
secure-code control.
```

This is not a circuit-level causal analysis, but it is a direct activation-space
analysis of the internal representations induced by persona prompts.

