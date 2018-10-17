#!/bin/bash

# run.sh

# cd data
# python make-data.py
# cd ..

export GRAPHBLAS_PATH=$HOME/projects/davis/GraphBLAS
make clean
make

python data/make-random.py --num-rows 5 --num-cols 6 --density 1.0
./proj --X data/X.mtx --unweighted 1 # --proj-debug 1

