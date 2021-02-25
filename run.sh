#!/bin/bash

# run.sh

# cd data
# python make-data.py
# cd ..

export GRAPHBLAS_PATH=$(pwd)/GraphBLAS
make clean
make

python data/make-random.py --num-rows 5000 --num-cols 5000 --density 0.01
python data/mtx2bin.py --inpath data/X.mtx
python reference.py data/X.mtx

make clean ; make ; ./proj data/X.bin

# --

python data/mtx2bin.py --inpath data/ml_1000000.mtx
python reference.py data/ml_1000000.mtx
./proj data/ml_1000000.bin

python data/mtx2bin.py --inpath data/ml_5000000.mtx
python reference.py data/ml_5000000.mtx
./proj data/ml_5000000.bin

# --

python data/make-random.py --num-rows 10 --num-cols 10 --density 0.1
python data/mtx2bin.py --inpath data/X.mtx
python reference.py data/X.mtx

make clean ; make ; ./proj data/X.bin
make clean ; make ; ./proj data/ml_1000000.bin
make clean ; make ; ./proj data/ml_5000000.bin