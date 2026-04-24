# CS8803 DRL final-project report

LaTeX source for the report on `055v2_extend@1750` (the submitted Soccer-Twos
agent).  Compiled output is `main.pdf`.

## Layout

```
report/
├── main.tex                       LaTeX source (CoRL 2026 template)
├── main.bib                       BibTeX references (9 entries)
├── main.pdf                       compiled output (5 pages)
├── corl_2026.sty                  CoRL style file (vendored)
├── corlabbrvnat.bst               CoRL bibliography style (vendored)
│
├── figures/                       generated figure PDFs + PNGs
│   ├── architecture.pdf           Fig. 1 — Siamese + cross-attention policy
│   ├── pipeline.pdf               Fig. 2 — 3-stage distillation pipeline
│   └── training_curves.pdf        Fig. 3 — official 1000-ep eval over iters
│
├── html/                          HTML+SVG sources for Fig. 1 and Fig. 2
│   ├── architecture.html
│   └── pipeline.html
│
└── scripts/                       figure renderers
    ├── plot_training_curves.py    matplotlib (reads experiment logs)
    ├── plot_architecture.py       (legacy, superseded by html/)
    ├── plot_pipeline.py           (legacy, superseded by html/)
    └── render_html_figs.sh        renders html/*.html → figures/*.pdf via headless Chrome
```

## Build

The HTML figures are rendered to PDF via headless Chromium; the matplotlib
plot reads experiment logs from `../docs/experiments/artifacts/official-evals/`.
The LaTeX is compiled with [Tectonic](https://tectonic-typesetting.github.io/)
(self-bootstrapping, no system TeX required).

```bash
# 1. (Re)render figures
bash scripts/render_html_figs.sh                                    # → figures/architecture.pdf, pipeline.pdf
python scripts/plot_training_curves.py                              # → figures/training_curves.pdf

# 2. Compile
~/.local/bin/tectonic main.tex                                      # → main.pdf
```

## Notes

- All three figures are rendered natively at the CoRL text column width
  (5.5 in) so that `\includegraphics[width=\linewidth]{...}` embeds at 1:1
  with no font shrinkage.
- The `html/` figures use Inter + JetBrains Mono throughout; the matplotlib
  plot uses the same `font.family` for visual consistency.
- The bibliography is auto-trimmed to only cited entries; unused entries in
  `main.bib` are silently dropped at compile time.
