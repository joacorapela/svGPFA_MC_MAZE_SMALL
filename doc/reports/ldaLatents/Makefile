all: report30190367.pdf

%.pdf: %.tex
	pdflatex $<
	bibtex $*
	pdflatex $<
	pdflatex $<

clean:
	rm report30190367.aux report30190367.bbl report30190367.blg report30190367.lof report30190367.log report30190367.out
