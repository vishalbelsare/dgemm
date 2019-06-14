import pytest
import numpy as np
import dgemmPy as dg


def equality(N, M, K):
    np.random.seed(1)
    a = np.random.rand(M, K)
    b = np.random.rand(K, N)
    c1 = dg.dgemm_py_blas(a, b)
    c2 = dg.sgemm_cuda_cublas(a, b)
    diff = np.absolute(c1 - c2)
    return np.sum(diff)


def test_equality_6():
    N = 40
    M = 50
    K = 30
    assert equality(N, M, K) == pytest.approx(0.0, abs=0.1)

