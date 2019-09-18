#!/bin/sh

jupyter nbconvert ../examples/example_constant.ipynb --to rst 
mv ../examples/example_constant.rst ./source/
rm ./source/example_constant_files -r
mv ../examples/example_constant_files ./source/ -f

make html

