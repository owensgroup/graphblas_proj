import sys
from scipy.io import mmread

inpath = sys.argv[1]
# inpath = 'data/X.mtx'

x = mmread(inpath).tocsr()

print(x.indptr[:10])
print(x.indices[:10])
print(x.data[:10])

p = x.T @ x
p.sort_indices()
print('--')
print(p.shape)
print(p.indptr[:10])
print(p.indices[:10])
print(p.data[:10])
print(p.nnz)

# p = x @ x.T
# p.sort_indices()
# print('--')
# print(p.shape)
# print(p.indptr[:10])
# print(p.indices[:10])
# print(p.data[:10])
# print(p.nnz)
