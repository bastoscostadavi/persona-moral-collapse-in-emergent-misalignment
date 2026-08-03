# Vendored from llm-persona-moral-metrics

This directory used to be a git submodule. It is now ordinary tracked files in
the parent repository, so the paper repo is self-contained and there is no
submodule pointer to dangle or diverge.

## Provenance

- Upstream: https://github.com/bastoscostadavi/llm-persona-moral-metrics
- Forked from upstream commit `5fdc03505f3af7c666eed22a63e4f36cf2af1a5a` ("updated readme")
- Local commit carried in: `1f65c05` ("Add thread pool to MFQ sampling; register trajectory checkpoints"),
  also pushed upstream as branch `mfq-sampling-thread-pool` so it is not lost.

At the time of vendoring, upstream `main` was 6 commits ahead of the
fork point with independent work on the *prior* paper (Oxford Utilitarianism
Scale, prompt-paraphrase robustness, persona-set sensitivity, rebuttal figures).
None of those commits touch the modules this repo depends on, so nothing was
dropped by not merging them.

## What differs from upstream

- `data/` and `paper/` were removed. The parent repo owns them now: sampling
  CSVs live in the repo-root `data/`, the manuscript in `submissions/`.
- `run_mfq_sampling.py` gained a thread pool (`--workers`); see the commit above.
- `config/models.yaml` carries the fine-tuned and checkpoint variants used by
  this paper, including a fenced block written by
  `checkpoint-trajectory/register_models.py`.
- `llm_interface.py`, `model_registry.py`, `analysis/compute_metrics.py`,
  `run_mfq_logits.py` and `config/benchmark.yaml` carry this paper's extensions
  (extra providers, models, and metrics options).

## If you need upstream changes

Pull them by hand against the upstream repo rather than re-adding a submodule;
that is the tradeoff this vendoring accepted deliberately.
