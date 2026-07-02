# NeurIPS 2026 Snapshot

Frozen artifacts for the NeurIPS 2026 paper version.

- `main.tex`, `checklist.tex`, `references.bib`, `neurips_2026.sty`,
  `figures/`, and `main.pdf` are the frozen NeurIPS paper snapshot.

Build command:

```bash
cd submissions/2026-neurips
latexmk -pdf main.tex
```

Figures are copied into this snapshot for reproducibility. Continue generating
current figures into the active `paper/figures/` directory at the repo root.
