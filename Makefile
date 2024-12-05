# Input variables
PYTHON_VERSION=3.11
MANAGER=micromamba
ENV_NAME=assaiku
PYTHON_FILES=assaiku
AUTHOR=ArnaudGuibbert
VERSION=$(shell cat VERSION)

# Custom variables
RUN_CMD=$(MANAGER) run -n $(ENV_NAME)

include ./makefiles/hub.Makefile
include ./makefiles/python.Makefile
include ./makefiles/docs.Makefile