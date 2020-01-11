#!/bin/sh

rm ./source/example_constant_files/*
jupyter nbconvert ../examples/example_constant.ipynb --to rst 
mv ../examples/example_constant.rst ./source/
mv ../examples/example_constant_files/* ./source/_static
sed -i 's/example_constant_files/_static/g' source/example_constant.rst

# jupyter nbconvert ../examples/symenergy_doc_cookbook.ipynb --to rst 
# mv ../examples/symenergy_doc_cookbook.rst ./source/
# mv ../examples/symenergy_doc_cookbook_files ./source/

make html

