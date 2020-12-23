PY = python3
RM = rm -rf
TWINE = twine
TOX = tox
LINT = pylint --rcfile=./.pylintrc

.PHONY: all check dist sdist test tox tox-v tox-vv tox-report lint doc upload clean

all: dist check test

dist: sdist bdist_wheel

test: tox lint

sdist bdist_wheel:
	$(PY) setup.py $@

lint:
	$(LINT) ckip_transformers

check:
	$(TWINE) check dist/*

tox tox-v tox-vv tox-report:
	( cd test && make $@ )

doc:
	( cd docs && make clean && make html )

upload: dist check
	ls dist/*
	$(TWINE) upload --repository-url https://test.pypi.org/legacy/ dist/* --verbose

clean:
	- ( cd docs && make clean )
	- ( cd test && make clean )
	- $(PY) setup.py clean -a
	- $(RM) build dist *.egg-info __pycache__
