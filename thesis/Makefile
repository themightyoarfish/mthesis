TEX_TMEPLATE=template.tex
OUT=thesis
IN_FILES=$(shell ls chapters/*)
PANDOC_ARGS=-f markdown+raw_tex --top-level-division=chapter --filter pandoc-crossref --listings --filter pandoc-citeproc

all: $(OUT).pdf
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
texpdf: $(OUT).pdf

# Generate pdf directly from markdown
pdf: $(TEX_TMEPLATE) $(IN_FILES)
	pandoc $(PANDOC_ARGS) --template $(TEX_TMEPLATE) -o $(OUT).pdf $(IN_FILES)

open: $(OUT).pdf
	open $<

.PHONY: clean

clean:
	rm $(OUT).*
	rm texput.log
