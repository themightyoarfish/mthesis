MAIN_FILE=Thesis
TEX_SOURCES=$(MAIN_FILE).tex Chapters/*.tex classicthesis-config.tex FrontBackMatter/*.tex Bibliography.bib
TEX_CMD="latexmk -pdf"
all: $(MAIN_FILE).pdf

$(MAIN_FILE).pdf: $(TEX_SOURCES) FrontBackMatter/Titlepage.pdf
	$(TEX_CMD) $(MAIN_FILE).tex
	open -g -a Skim "$(MAIN_FILE).pdf"

FrontBackMatter/Titlepage.pdf: FrontBackMatter/Titlepage.tex gfx/uni.pdf
	cd FrontBackMatter; $(TEX_CMD) Titlepage.tex; cd ..;

clean:
	latexmk -c
	cd FrontBackMatter && latexmk -c && cd ..
	cd Chapters && latexmk -c && cd ..
