.PHONY: py-format
py-format:
	$(RUN_CMD) ruff format $(PYTHON_FILES)

.PHONY: py-lint
py-lint:
	$(RUN_CMD) ruff check --fix $(PYTHON_FILES)

.PHONY: py-structure
py-structure:
	$(MAKE) -k py-format py-lint

.PHONY: py-analyze
py-analyze:
	$(RUN_CMD) mypy $(PYTHON_FILES)

.PHONY: py-test
py-test:
	$(RUN_CMD) pytest tests

.PHONY: py-check
py-check:
	$(RUN_CMD) ruff format --check
	$(RUN_CMD) ruff check
	$(MAKE) py-test
	