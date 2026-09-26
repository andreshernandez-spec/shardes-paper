# Build the paper. `make tables` re-emits every generated table from its
# script; `make` builds the PDF, `make anon` the double-blind one. Tables are
# never edited by hand.
LATEX=latexmk -pdf -interaction=nonstopmode -halt-on-error

all: tables
	$(LATEX) main.tex

# Double-blind build: no author, no acknowledgements, no repository URLs.
anon: tables
	$(LATEX) -jobname=main-anon -usepretex='\def\anonymousbuild{}' main.tex

tables:
	cd ../experiments/phase2 && python tb1.py --latex >/dev/null
	cd ../experiments/phase2 && python tb3.py --latex >/dev/null
	cd ../experiments/phase2 && python tb8.py --latex >/dev/null
	cd ../experiments/phase2/multihost && python tb7_e18.py --latex >/dev/null
	cd ../experiments/countdown && python tb5_e17.py --latex >/dev/null
	cd ../experiments/countdown && python tb6_e16.py --latex >/dev/null

clean:
	latexmk -C main.tex
	latexmk -C -jobname=main-anon main.tex
