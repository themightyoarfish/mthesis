MAIN_FILE=Thesis
TEX_SOURCES=$(MAIN_FILE).tex Chapters/*.tex classicthesis-config.tex FrontBackMatter/*.tex Bibliography.bib
TEX_CMD=pdflatex -shell-escape
SPEED_FLAGS=-draftmode --interaction=batchmode

all: $(MAIN_FILE).pdf

$(MAIN_FILE).pdf: $(TEX_SOURCES) FrontBackMatter/Titlepage.pdf
	@echo "Initial draft..."
	$(TEX_CMD) -draftmode "$(MAIN_FILE).tex"
	@echo "Running bibtex..."
	bibtex "$(MAIN_FILE)"
	@echo "Compiling document..."
	$(TEX_CMD) $(SPEED_FLAGS) "$(MAIN_FILE).tex" > /dev/null 2>&1
	$(TEX_CMD) --interaction=batchmode "$(MAIN_FILE).tex" > /dev/null 2>&1
	open -g -a Skim "$(MAIN_FILE).pdf"

once: $(TEX_SOURCES) FrontBackMatter/Titlepage.pdf
	$(TEX_CMD) "$(MAIN_FILE).tex";
	open -g -a Skim "$(MAIN_FILE).pdf"

FrontBackMatter/Titlepage.pdf: FrontBackMatter/Titlepage.tex gfx/uni.pdf
	cd FrontBackMatter; $(TEX_CMD) Titlepage.tex; cd ..;

clean:
	$(RM) $(MAIN_FILE).pdf *.aux *.log Chapters/*.aux Chapters/*.log FrontBackMatter/*.aux FrontBackMatter/*.log *.auxlock *.toc *.bbl *.blg *.out *.lof *.lot
