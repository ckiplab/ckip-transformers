RM = rm -rf
TOX = tox

.PHONY: tox tox-v tox-report clean

tox:
	NO_COV= $(TOX) -p -e py36-hf{3,4} --

tox-v:
	NO_COV= $(TOX) -e py36-hf{3,4} -- -v

tox-vv:
	NO_COV= $(TOX) -e py36-hf4 -- -vv

tox-report:
	- $(TOX) -p -e clean,py36-hf4,report -- --cov-append
	python3.7 -m http.server --directory .test/htmlcov/ 3000

clean:
	- $(RM) .tox .test .lookup
