from __future__ import absolute_import
import timeit
import numpy as np
import dgemmPy as dg


np.random.seed(1)

M = 40
N = 500
K = 52

a = np.random.rand(M, K)
b = np.random.rand(K, N)

c1 = dg.dgemm_py_blas(a, b)
c2 = dg.dgemm_py_loops(a, b)

print(np.sum(c1 - c2))

repeats = 10
global_repeats = 20

print("py_blas:                      {:f} ms".format(
    timeit.timeit(stmt="dg.dgemm_py_blas(a, b, repeats)",
                  number=global_repeats,
                  globals=globals()) * 1e3 / global_repeats))
print("py_loops:                     {:f} ms".format(
    timeit.timeit(stmt="dg.dgemm_py_loops(a, b, repeats)",
                  number=global_repeats,
                  globals=globals()) * 1e3 / global_repeats))
