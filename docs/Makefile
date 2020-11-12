# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINX     ?= sphinx-build
SPHINXOPTS ?=
DOCDIR     = .
BUILDDIR   = _build
APIDIR     = _api

# Remember add the environment variable to readthedocs.org!!!
ENV = SPHINX_APIDOC_OPTIONS=members,show-inheritance

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINX) -M help "$(DOCDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean serve

clean: Makefile
	@$(SPHINX) -M $@ "$(DOCDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@$(SPHINX) -M $@ "$(DOCDIR)" "$(APIDIR)"   $(SPHINXOPTS)

serve:
	( cd $(BUILDDIR)/html/ && python3 -m http.server 8080 )

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(ENV) $(SPHINX) -M $@ "$(DOCDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
