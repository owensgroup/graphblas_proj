#!/usr/bin/env python

"""
    random-mtx.py
"""

import sys
import argparse
import numpy as np
from scipy import sparse
from scipy.io import mmwrite

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-rows', type=int, default=4096)
    parser.add_argument('--num-cols', type=int, default=2048)
    parser.add_argument('--density', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    X = sparse.random(args.num_rows, args.num_cols, args.density)
    X.data = np.ones(X.nnz, dtype=np.float32)
    mmwrite('data/X', X)
