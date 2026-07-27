# Submissions

This directory holds one self-contained folder per venue submission. Each folder
carries its own paper source, style files, `references.bib`, and a local copy of
every figure it uses, so it can be built in isolation without depending on
anything elsewhere in the repo.

- `2026-neurips/` - NeurIPS 2026 paper. This is the active paper version; it also
  keeps peer-review files under `reviews/`.
- `2026-icml/` - LatinX in AI Research Workshop at ICML 2026, with `poster/` and
  `oral-presentation/` assets.

Figures are intentionally duplicated inside each submission folder so that each
one preserves the exact visual state used for that venue, even as the project's
analysis scripts continue to regenerate results.

Build (example):

```bash
cd submissions/2026-neurips
latexmk -pdf main.tex
```
