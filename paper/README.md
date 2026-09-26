# paper/

The MLSys-style paper. `main.tex` holds the abstract and the section order; `sections/`
has one file per section, in reading order. `generated/` holds the tables, written by
`experiments/phase2/tb1.py`, `tb3.py`, `tb8.py`, `multihost/tb7_e18.py` and
`experiments/countdown/tb5_e17.py`, `tb6_e16.py`, and is never edited by hand
(`make tables`; CI fails if a regenerated table changes a byte). Figures are included
straight from `experiments/*/figures`, so regenerating a figure updates the paper on the
next build.

    make          # tables and figures, then main.pdf
    make anon     # tables and figures, then main-anon.pdf: no author, acknowledgements or experiments URL

Swap the document class for the official MLSys one when the style is in.

`make figures` regenerates the main placement, common-configuration cost, real-model,
and learning plots, including the full cost grid and frozen-embedding appendix plot.
`make` and `make anon` run both table and figure generation before LaTeX. Set `PYTHON`
to the intended interpreter when the active shell is outside the project environment.
The full scaling grids and synthetic-alignment plot retain their original generators.
