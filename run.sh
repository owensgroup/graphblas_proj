#!/bin/bash

# run.sh

# cd data
# python make-data.py
# cd ..

export GRAPHBLAS_PATH=$HOME/projects/davis/GraphBLAS
make clean
make

python data/make-random.py --seed 111 --num-rows 1000 --num-cols 1000 --density 0.001
./proj --X data/X.mtx --unweighted 1 --proj-debug 1


./proj --X small_ratings.mtx --unweighted 1 --proj-debug 1 --num-chunks 0
./proj --X small_ratings.mtx --unweighted 1 --proj-debug 1 --num-chunks 0 --onto-cols 0


./proj --X graph500-scale18-ef16_adj.mtx --unweighted 1 --proj-debug 1 --num-chunks 3

X=../hive_tests/proj/ml-20m/ratings.mtx
./proj --X $X --unweighted 1 --proj-debug 1 --num-chunks 10 --onto-cols 0