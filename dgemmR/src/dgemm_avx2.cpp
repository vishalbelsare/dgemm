#include "dgemm.h"

void dgemm::dgemm_C_loops_avx2(double* aligned_a,
                               double* aligned_b,
                               double* aligned_c,
                               int M,
                               int K,
                               int N,
                               int repeats) {
    double *_a, *_b, *_c;
    __m256d a, b, c;
    for (int r = 0; r < repeats; r++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                _a = &aligned_a[m * K];  // INDEX_ROW(m, 0, M, K)
                                         // = m * K
                _b = &aligned_b[n * K];  // INDEX_COL(0, n, K,
                                         // N) = n * K

                c = _mm256_setzero_pd();
                for (int k = 0; k < K; k += 4) {
                    a = _mm256_load_pd(&_a[k]);
                    b = _mm256_load_pd(&_b[k]);
                    c = _mm256_fmadd_pd(a, b, c);  // c = a * b + c
                }

                _c = (double*)&c;
                aligned_c[INDEX(m, n, M, N)] = _c[0] + _c[1] + _c[2] + _c[3];
            }
        }
    }
}
