TEX_TMEPLATE=template.tex
OUT=thesis
IN_FILES=$(shell ls chapters/*)
PANDOC_ARGS=--top-level-division=chapter

all: pdf
	open $(OUT).pdf

# Generate latex file
$(OUT).tex: $(TEX_TMEPLATE) $(IN_FILES)
	pandoc $(PANDOC_ARGS) --template $(TEX_TMEPLATE) -o $(OUT).tex $(IN_FILES)

# compile latex to pdf
$(OUT).pdf: $(OUT).tex
	pdflatex -jobname $(OUT) $<

# shortcut for tex creation
tex: $(OUT).tex

# shortcut for pdf creation
pdf: $(OUT).pdf

.PHONY: clean

clean:
	rm $(OUT).*
	rm texput.log
