#!/usr/bin/env python

"""
  reference.py
"""

import sys
from scipy.io import mmread

inpath = sys.argv[1]

x = mmread(inpath)
x = x.tocsr()

p = x.T @ x
print(p.nnz)