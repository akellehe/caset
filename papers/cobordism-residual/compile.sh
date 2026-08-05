#!/bin/bash
# Build the draft (clean: 0 errors, 0 overfull boxes, 0 Type 3 fonts).
#
# The directory vendors ltxgrid.sty/ltxutil.sty from revtex 4.2f (TeX Live
# trunk) because quantumarticle needs them and this box lacks
# texlive-publishers; on a machine with a full revtex install they are
# unnecessary and can be deleted. main.tex loads lmodern *before*
# \documentclass: quantumarticle only loads it \AtBeginDocument, which is too
# late for ltxgrid's saved column font state here, and the two-column body
# silently falls back to bitmap EC (Type 3) fonts without it.
#
# fig-gate-battery.pdf regenerates from a battery run log via
#   python fig-gate-battery.py /path/to/run.log
# (TrueType-embedded, no Type 3 fonts).
set -e
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
