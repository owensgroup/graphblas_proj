import sys
from scipy.io import mmread

inpath = sys.argv[1]
# inpath = 'data/X.mtx'

x = mmread(inpath).tocsr()

p = x @ x
print(p.nnz)

p = x @ x.T
print(p.nnz)

p = x.T @ x
print(p.nnz)
print(p.todense())