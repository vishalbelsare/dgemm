#include "dgemm.h"

void dgemm::dgemm_C_loops(double* matrix_a,
                          double* matrix_b,
                          double* result,
                          int M,
                          int K,
                          int N,
                          int repeats) {
    double sum;

    for (int r = 0; r < repeats; r++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                sum = 0.0;
                for (int k = 0; k < K; k++) {
                    sum += matrix_a[INDEX(m, k, M, K)] *
                           matrix_b[INDEX(k, n, K, N)];
                }
                result[INDEX(m, n, M, N)] = sum;
            }
        }
    }
}

