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
c3 = dg.dgemm_C_loops(a, b)
c4 = dg.dgemm_C_blas(a, b)
c5 = dg.dgemm_C_loops_avx(a, b)
c6 = dg.dgemm_C_loops_avx_omp(a, b)
c7 = dg.dgemm_C_loops_avx_tp(a, b)

print(np.sum(c1 - c2))
print(np.sum(c1 - c3))
print(np.sum(c1 - c4))
print(np.sum(c1 - c5))
print(np.sum(c1 - c6))
print(np.sum(c1 - c7))

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
print("C_loops:                      {:f} ms".format(
    timeit.timeit(stmt="dg.dgemm_C_loops(a, b, repeats)",
                  number=global_repeats,
                  globals=globals()) * 1e3 / global_repeats))
print("C_blas:                       {:f} ms".format(
    timeit.timeit(stmt="dg.dgemm_C_blas(a, b, repeats)",
                  number=global_repeats,
                  globals=globals()) * 1e3 / global_repeats))
print("C_loops_avx:                  {:f} ms".format(
    timeit.timeit(stmt="dg.dgemm_C_loops_avx(a, b, repeats)",
                  number=global_repeats,
                  globals=globals()) * 1e3 / global_repeats))
print("C_loops_avx_omp:              {:f} ms".format(
    timeit.timeit(stmt="dg.dgemm_C_loops_avx_omp(a, b, repeats)",
                  number=global_repeats,
                  globals=globals()) * 1e3 / global_repeats))
print("C_loops_avx_tp:               {:f} ms".format(
    timeit.timeit(stmt="dg.dgemm_C_loops_avx_tp(a, b, repeats)",
                  number=global_repeats,
                  globals=globals()) * 1e3 / global_repeats))
