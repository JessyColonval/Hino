TARGET=main

TMP=$(wildcard *.aux) $(wildcard *.bbl) $(wildcard *.blg) $(wildcard *.log) $(wildcard *.toc) $(wildcard *.out)

$(TARGET).pdf: $(TARGET).tex
	pdflatex --shell-escape $^
	pdflatex --shell-escape $^
	pdflatex --shell-escape $^

tar: *.tex Makefile *.pdf *.png *.sty
	tar -czvf PriseDeNotes.tar.gz $^

clean:
	rm -f $(TMP)

mrproper: clean
	rm -f $(TARGET).pdf
