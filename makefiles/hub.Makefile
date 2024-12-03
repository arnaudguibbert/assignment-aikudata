.PHONY: py-init-env
py-init-env:
	$(MANAGER) create -y -n $(ENV_NAME) python=$(PYTHON_VERSION)

.PHONY: py-install-dev
py-install-dev:
	$(RUN_CMD) pip install -U -e .[dev]

.PHONY: py-install-full
py-install-full:
	$(RUN_CMD) pip install -U -e .[dev,docs]

.PHONY: py-install
py-install:
	$(RUN_CMD) pip install -U .

.PHONY: py-init-kernel
py-init-kernel:
	$(RUN_CMD) pip install ipykernel ipywidgets
	$(RUN_CMD) python -m ipykernel install \
		--user --name $(ENV_NAME) \
		--display-name "Python $(PYTHON_VERSION) ($(ENV_NAME))"
    
.PHONY: py-setup-env-dev
py-setup-env-dev:
	$(MAKE) py-init-env
	$(MAKE) py-install-dev
	$(MAKE) py-init-kernel