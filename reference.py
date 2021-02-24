from time import time
import numpy as np
from scipy import sparse
from scipy.io import mmwrite

from rsub import *
from matplotlib import pyplot as plt

calls = np.load('./data/calls.npy')

num_nodes = calls.max() + 1
# np.unique(np.hstack(calls)).shape

data = np.ones(calls.shape[0])
rows = calls[:,0]
cols = calls[:,1]

m = sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))

sel = np.random.permutation(num_nodes)
m = m[sel][:,sel]

t = time()
N = 300000
z = m[:N,:N]
p = z.T.dot(z)
time() - t

mmwrite('./data/calls', z, field='pattern', symmetry='general')

(p.data.nbytes + p.indptr.nbytes + p.indices.nbytes) / 1e9

# --

from scipy.io import mmread
from time import time

x = mmread('small_ratings.mtx')
x = x.tocsr()

t = time()
p = x.T.dot(x)
p.nnz
time() - t

t = time()
p = x.dot(x.T)
p.nnz
time() - t