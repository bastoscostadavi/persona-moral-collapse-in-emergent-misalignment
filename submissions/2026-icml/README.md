# ICML 2026 Workshop Snapshot

Frozen artifacts for the LatinX in AI Research Workshop at ICML 2026.

- `main.tex`, `references.bib`, style files, `figures/`, and `main.pdf` are the
  frozen ICML paper snapshot.
- `poster/` contains the poster source, poster figures/assets, compiled PDF,
  upload PNG, and thumbnail PNG.

Build commands:

```bash
cd submissions/2026-icml
latexmk -pdf main.tex

cd poster
latexmk -pdf persona_model_collapse_poster.tex
```

The poster bibliography resolves through `../references.bib`, so keep the poster
folder with this submission snapshot.
