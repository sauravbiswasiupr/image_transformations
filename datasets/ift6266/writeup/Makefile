all: aistats2011_cameraready.pdf
#all: nips2010_submission.pdf

%.pdf: %.tex images/* *.bib
	pdflatex $<
	bibtex -min-crossrefs=999 ${<:.tex=}
	pdflatex $<
	pdflatex $<
	pdflatex $<

clean:
	rm -f *.aux *.bbl *.blg *.log