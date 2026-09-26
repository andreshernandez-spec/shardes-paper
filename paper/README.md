# Paper builds

The build needs the Python environment in the repository requirements, a TeX Live
installation with `latexmk`, and Poppler's `pdftotext` for the submission check.

- `make`: named, combined reading copy, `main.pdf`.
- `make anon`: anonymous combined reading copy, `main-anon.pdf`.
- `make submission`: anonymous `main-submission.pdf` and `appendix-anon.pdf` for
  separate uploads. References, including those cited in the appendix, are in the
  main file. Appendix citations are plain text; section, figure and table references
  link to their companion PDF when both files are kept together. A PDF viewer or
  submission platform may restrict links between files.

The combined `submission-all.pdf` supplies shared labels, bibliography and numbering;
it is an intermediate, not the main upload. `check_submission.py` checks the body page
limit and visible identifying text. The anonymous variants withhold the library name
and repository URLs. No anonymous repository mirror or submission is created by the build.

The [MLSys 2027 research CFP](https://mlsys.org/Conferences/2027/CallForResearchPapers)
uses the 2025 style, allows ten body pages, requires all authors in each reference,
and requires a separate appendix upload. The five vendored files `mlsys2025.sty`,
`mlsys2025.bst`, `fancyhdr.sty`, `algorithm.sty`, and `algorithmic.sty` are unchanged from
[the linked official archive](https://media.mlsys.org/Conferences/MLSYS2025/mlsys2025style.zip),
retrieved 2026-09-26. Their original license and attribution notices are retained.

`main.tex` holds the section order; `sections/` contains the abstract and body files.
Tables in `generated/` are written by the committed generators, never edited by hand;
CI checks that `make tables` reproduces them byte for byte. Figures are included directly
from `experiments/*/figures`.

`make figures` regenerates the main plots plus the full cost and frozen-embedding plots.
The full scaling grids and synthetic-alignment plot retain their original generators.
All PDF targets regenerate tables and these figures first. Set `PYTHON` to the intended
interpreter when the active shell is outside the project environment.
`experiments/phase2/paper_evidence.py` prints the worked-model, provenance, communication,
matched-precision and learning-distance checks. Qwen timing tables and plots exclude
records whose `dirty_worktree` flag is true or missing; the historical records remain
unchanged. The results directory documents which comparisons this removes.
