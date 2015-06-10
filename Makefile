rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))

CC := gcc
CXX := g++
CFLAGS := -std=c++11 -shared -fPIC

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
        CXX := clang
	CFLAGS += --stdlib=libc++ -undefined dynamic_lookup
endif

SRCS := $(call rwildcard, sptgraph/, *.cpp)
TARGETS := $(SRCS:%.cpp=%.so)


.PHONY: clean-pyc clean-build docs clean all

all: $(TARGETS)

%.so: %.cpp
	$(CXX) -o $@ $(CFLAGS) -I $(GRAPHLAB_SDK_HOME) $^


help:
	@echo "clean - remove all build, test, coverage and Python artifacts"
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "clean-test - remove test and coverage artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "test-all - run tests on every Python version with tox"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "docs - generate Sphinx HTML documentation, including API docs"
	@echo "release - package and upload a release"
	@echo "dist - package"
	@echo "all - build cpp extension for graphlab"

clean: clean-build clean-pyc clean-test clean-so

clean-so:
	find . -name '*.so' -exec rm -f {} +

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/

lint:
	flake8 sptgraph tests

test:
	python setup.py test

test-all:
	tox

coverage:
	coverage run --source sptgraph setup.py test
	coverage report -m
	coverage html
	open htmlcov/index.html

docs:
	rm -f docs/sptgraph.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ sptgraph
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	open docs/_build/html/index.html

release: clean
	python setup.py sdist upload
	python setup.py bdist_wheel upload

dist: clean
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist
