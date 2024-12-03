.PHONY: init-docs
init-docs:
	mkdir -p docs
	sphinx-quickstart docs \
		--sep \
		--ext-autodoc \
		-a $(AUTHOR) \
		-p $(ENV_NAME) \
		-r $(VERSION) \
		-l en \
		
.PHONY: auto-docs
auto-docs:
	sphinx-apidoc \
		--output-dir docs/source \
		$(ENV_NAME)
	echo "	modules" >> docs/source/index.rst

.PHONY: build-docs
build-docs:
	cd docs && $(MAKE) clean
	cd docs && $(MAKE) html

.PHONY: delete-docs
delete-docs:
	rm -rf docs