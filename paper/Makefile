# Build the paper. `make tables` re-emits every generated table from its
# script; `make` builds the PDF, `make anon` the double-blind one. Tables are
# never edited by hand.
PYTHON ?= python

.PHONY: all anon submission tables figures clean

LATEX=latexmk -pdf -interaction=nonstopmode -halt-on-error

all: tables figures
	$(LATEX) main.tex

# Double-blind build: no author, no acknowledgements, no repository URLs.
anon: tables figures
	$(LATEX) -jobname=main-anon -usepretex='\def\anonymousbuild{}' main.tex

# The official 2027 CFP reuses the 2025 style and requires a separate appendix.
# Compile once for shared reference data, then produce the two upload files.
submission: tables figures
	$(LATEX) -jobname=submission-all -usepretex='\def\anonymousbuild{}\def\submissionbuild{}' main.tex
	$(PYTHON) prepare_submission.py
	$(LATEX) -jobname=main-submission -usepretex='\def\anonymousbuild{}\def\submissionbuild{}\def\submissionpart{}' main.tex
	$(LATEX) -jobname=appendix-anon -usepretex='\def\anonymousbuild{}\def\submissionbuild{}\def\submissionpart{}\def\appendixonly{}' main.tex
	$(PYTHON) check_submission.py

tables:
	cd ../experiments/phase2 && $(PYTHON) tb1.py --latex >/dev/null
	cd ../experiments/phase2 && $(PYTHON) tb3.py --latex >/dev/null
	cd ../experiments/phase2 && $(PYTHON) tb8.py --latex >/dev/null
	cd ../experiments/phase2 && $(PYTHON) paper_evidence.py --latex >/dev/null
	cd ../experiments/phase2/multihost && $(PYTHON) tb7_e18.py --latex >/dev/null
	cd ../experiments/countdown && $(PYTHON) tb5_e17.py --latex >/dev/null
	cd ../experiments/countdown && $(PYTHON) tb6_e16.py --latex >/dev/null

figures:
	cd ../experiments/phase2 && $(PYTHON) plot_paper.py >/dev/null
	cd ../experiments/phase2 && $(PYTHON) plot_cost.py --results results-cost results-cost-tpu-v5e8 >/dev/null
	cd ../experiments/countdown && $(PYTHON) plot_e13.py >/dev/null
	cd ../experiments/countdown && $(PYTHON) plot_e17.py >/dev/null

clean:
	latexmk -C main.tex
	latexmk -C -jobname=main-anon main.tex
	latexmk -C -jobname=submission-all main.tex
	latexmk -C -jobname=main-submission main.tex
	latexmk -C -jobname=appendix-anon main.tex
	rm -f submission-counters.tex submission-main-refs.aux submission-appendix-refs.aux
