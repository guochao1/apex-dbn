all: test

apex_exp_template: apex_exp_template.cpp apex_exp_template.h
	g++ -o apex_exp_template apex_exp_template.cpp

doc:
	doxygen apex_exp_template.cpp

test: test.cpp
	g++ -o test test.cpp