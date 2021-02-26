#!/bin/bash

# run.sh

# --
# Grab data

cd data
wget https://graphchallenge.s3.amazonaws.com/synthetic/graph500-scale18-ef16/graph500-scale18-ef16_adj.mmio.gz
gunzip graph500-scale18-ef16_adj.mmio.gz
cd ..

# --
# Prep data

python data/mtx2bin.py --inpath data/ml_1000000.mtx
python data/mtx2bin.py --inpath data/ml_5000000.mtx
python data/mtx2bin.py --inpath data/ml_full.mtx
python data/mtx2bin.py --inpath data/graph500-scale18-ef16_adj.mmio

# --
# Reference results

python reference.py data/ml_1000000.mtx
# 63104132
python reference.py data/ml_5000000.mtx
# 157071858
python reference.py data/ml_full.mtx
# 286857534
python reference.py data/graph500-scale18-ef16_adj.mmio
# 

# --
# (m)GPU results

make clean ; make

CUDA_VISIBLE_DEVICES=0       ./proj data/ml_1000000.bin
CUDA_VISIBLE_DEVICES=1,2     ./proj data/ml_1000000.bin
CUDA_VISIBLE_DEVICES=0,1,2,3 ./proj data/ml_1000000.bin

CUDA_VISIBLE_DEVICES=0       ./proj data/ml_5000000.bin
CUDA_VISIBLE_DEVICES=1,2     ./proj data/ml_5000000.bin
CUDA_VISIBLE_DEVICES=0,1,2,3 ./proj data/ml_5000000.bin

CUDA_VISIBLE_DEVICES=0       ./proj data/ml_full.bin
CUDA_VISIBLE_DEVICES=1,2     ./proj data/ml_full.bin
CUDA_VISIBLE_DEVICES=0,1,2,3 ./proj data/ml_full.bin

CUDA_VISIBLE_DEVICES=0       ./proj data/graph500-scale18-ef16_adj.bin
CUDA_VISIBLE_DEVICES=1,2     ./proj data/graph500-scale18-ef16_adj.bin
CUDA_VISIBLE_DEVICES=0,1,2,3 ./proj data/graph500-scale18-ef16_adj.bin

# num_gpus = 1 | dataset = ml_1000000 | elapsed_ms = 161.17
# num_gpus = 2 | dataset = ml_1000000 | elapsed_ms = 110.943
# num_gpus = 4 | dataset = ml_1000000 | elapsed_ms = 89.3102

# num_gpus = 1 | dataset = ml_5000000 | elapsed_ms = 957.532
# num_gpus = 2 | dataset = ml_5000000 | elapsed_ms = 614.04
# num_gpus = 4 | dataset = ml_5000000 | elapsed_ms = 474.131

# num_gpus = 1 | dataset = ml_full    | elapsed_ms = 4534.08
# num_gpus = 2 | dataset = ml_full    | elapsed_ms = 2812.38
# num_gpus = 4 | dataset = ml_full    | elapsed_ms = 2112.62

# num_gpus = 1 | dataset = graph500-scale18-ef16_adj | elapsed_ms = FAIL
# num_gpus = 2 | dataset = graph500-scale18-ef16_adj | elapsed_ms = FAIL
# num_gpus = 4 | dataset = graph500-scale18-ef16_adj | elapsed_ms = FAIL
