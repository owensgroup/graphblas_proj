import sys
from scipy.io import mmread

inpath = sys.argv[1]
# inpath = 'data/X.mtx'

x = mmread(inpath).tocsr()

xT = x.T.tocsr()
xT.sort_indices()

print(x.indptr)
print(x.indices)
print(x.data)

print(xT.indptr)
print(xT.indices)
print(xT.data)


p = x.T @ x
p.sort_indices()

# p = x @ x.T
# p.sort_indices()
# print('--')
# print(p.shape)
# print(p.indptr[:10])
# print(p.indices[:10])
# print(p.data[:10])
# print(p.nnz)
