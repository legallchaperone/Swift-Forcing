# Paper (ICML 2026 template)

- `main.tex`: top-level file. Switch between blind review, preprint, and camera-ready on the `\usepackage{icml2026}` line.
- `sections/`: one file per section.
- `macros.tex`: shared notation.
- `references.bib`: bibliography.
- `figures/`: figure files.
- `icml2026_reference/`: the unmodified official example paper and its PDF, for formatting rules.

Build with `make` (the PDF is `build/main.pdf`), or `make watch` to rebuild on every change. In VS Code, the LaTeX Workshop extension rebuilds on save and shows the PDF in a tab. Or upload this folder to Overleaf and set `main.tex` as the main document.

ICML rules to remember: 8 pages for the main body. References, the Impact Statement, and the appendix do not count. No author names or acknowledgements in the review version.
