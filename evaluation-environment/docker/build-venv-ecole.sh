#!/bin/bash

# Clean virtualenv setup
rm -rf venv
python -m venv venv
source venv/bin/activate

# Extract SCIP into virtualenv
cd venv
bash ../SCIPOptSuite-7.0.2-Linux-debian.sh --skip-license
cd ..

# Ecole build requirements
pip install --upgrade pip setuptools
pip install numpy==1.20.2 scipy==1.6.3

# Extract ecole source
rm -rf ecole
tar xf v0.6.0.tar.gz

# Build ecole against SCIP
cd ecole-0.6.0/
SCIP_DIR=$(pwd)/venv/lib/cmake/scip cmake -B build/
cmake --build build/ --parallel
pip install --use-feature=in-tree-build build/python
