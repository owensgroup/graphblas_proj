import sys
from scipy.io import mmread

# inpath = sys.argv[1]
inpath = 'data/X.mtx'

x = mmread(inpath).tocsr()

xT = x.T.tocsr()
print(xT.indptr[:10])

p = x.T @ x
print(p.nnz)

p = x @ x.T
print(p.nnz)