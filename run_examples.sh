#!/bin/bash

# run all example scripts
for s in $(ls example*.py) ; do
	echo Running $s:
	python $s
done
