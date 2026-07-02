# Submission Snapshots

This directory stores frozen submission-specific artifacts. The active project code,
analysis scripts, generated results, and current paper sources remain at the repo
root, especially `paper/` and `paper/figures/`.

Use this directory for reproducible records of what was submitted to each venue:

- `2026-icml/` - LatinX in AI / ICML 2026 workshop paper snapshot, with poster
  assets in `poster/`.
- `2026-neurips/` - NeurIPS 2026 paper snapshot.

Figures are intentionally duplicated inside each submission folder. The central
`paper/figures/` directory is the working figure target used by analysis scripts;
submission-local figure copies preserve the exact visual state used by that
submission, even if the active project changes later.
