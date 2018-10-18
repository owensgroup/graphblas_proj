#!/bin/bash

# run.sh

# cd data
# python make-data.py
# cd ..

export GRAPHBLAS_PATH=$HOME/projects/davis/GraphBLAS
make clean
make

python data/make-random.py --seed 111 --num-rows 1000 --num-cols 1000 --density 0.001
./proj --X data/X.mtx --unweighted 1 --proj-debug 1 --print-results 0

