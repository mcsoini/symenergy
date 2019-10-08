#!/bin/sh

jupyter nbconvert ../examples/example_constant.ipynb --to rst 
mv ../examples/example_constant.rst ./source/
mv ../examples/example_constant_files ./source/

make html

