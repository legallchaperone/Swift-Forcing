# Build the paper with latexmk. The PDF goes to build/main.pdf.
MAIN = main

.PHONY: all watch clean

all:
	latexmk -pdf $(MAIN)

# Rebuild on every file change (use with any PDF viewer that auto-reloads).
watch:
	latexmk -pdf -pvc -view=none $(MAIN)

clean:
	latexmk -C $(MAIN)
	rm -rf build
