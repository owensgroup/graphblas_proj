#!/bin/bash

# run.sh

# cd data
# python make-data.py
# cd ..

export GRAPHBLAS_PATH=$HOME/projects/davis/GraphBLAS
make clean
make

python data/make-random.py --num-rows 40000 --num-cols 40000 --density 0.01
./proj --X data/X.mtx --unweighted 1 --proj-debug 1 --print-results 0

