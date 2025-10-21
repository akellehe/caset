.PHONY: html clean doxygen

doxygen:
	@echo "==> Running Doxygen"
	doxygen Doxyfile

html: doxygen
	@echo "==> Building Sphinx HTML"
	sphinx-build -b html source _build/html

clean:
	rm -rf _doxygen _build
