#!/bin/bash

# run.sh

# cd data
# python make-data.py
# cd ..

export GRAPHBLAS_PATH=$HOME/projects/davis/GraphBLAS
make clean
make

# python data/make-random.py --num-seeds 100
./proj --X data/X.mtx --proj-debug 1

