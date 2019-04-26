#include "dgemm.h"

void dgemm::dgemm_C_loops_avx512(double* aligned_a,
                                 double* aligned_b,
                                 double* aligned_c,
                                 int M,
                                 int K,
                                 int N,
                                 int repeats) {
    double *_a, *_b, *_c;
    __m512d a, b, c;
    for (int r = 0; r < repeats; r++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                _a = &aligned_a[m * K];  // INDEX_ROW(m, 0, M, K)
                                         // = m * K
                _b = &aligned_b[n * K];  // INDEX_COL(0, n, K,
                                         // N) = n * K

                c = _mm512_setzero_pd();
                for (int k = 0; k < K; k += 8) {
                    a = _mm512_load_pd(&_a[k]);
                    b = _mm512_load_pd(&_b[k]);
                    c = _mm512_fmadd_pd(a, b, c);  // c = a * b + c
                }

                _c = (double*)&c;
                aligned_c[INDEX(m, n, M, N)] = _c[0] + _c[1] + _c[2] + _c[3] +
                                               _c[4] + _c[5] + _c[6] + _c[7];
            }
        }
    }
}

void dgemm::dgemm_C_loops_avx512_omp(double* aligned_a,
                                     double* aligned_b,
                                     double* aligned_c,
                                     int M,
                                     int K,
                                     int N,
                                     int repeats) {
#if defined _OPENMP
    size_t maxThreads = omp_get_max_threads();
    omp_set_num_threads(maxThreads);
#endif

    for (int r = 0; r < repeats; r++) {
#pragma omp parallel for
        for (int m = 0; m < M; m++) {
            double *_a, *_b, *_c;
            __m512d a, b, c;
            for (int n = 0; n < N; n++) {
                _a = &aligned_a[m * K];  // INDEX_ROW(m, 0, M, K)
                                         // = m * K
                _b = &aligned_b[n * K];  // INDEX_COL(0, n, K,
                                         // N) = n * K

                c = _mm512_setzero_pd();
                for (int k = 0; k < K; k += 8) {
                    a = _mm512_load_pd(&_a[k]);
                    b = _mm512_load_pd(&_b[k]);
                    c = _mm512_fmadd_pd(a, b, c);  // c = a * b + c
                }

                _c = (double*)&c;
                aligned_c[INDEX(m, n, M, N)] = _c[0] + _c[1] + _c[2] + _c[3] +
                                               _c[4] + _c[5] + _c[6] + _c[7];
            }
        }
    }
}

void dgemm::dgemm_C_loops_avx512_tp(double* aligned_a,
                                    double* aligned_b,
                                    double* aligned_c,
                                    int M,
                                    int K,
                                    int N,
                                    int repeats) {
    size_t maxThreads = std::thread::hardware_concurrency();
    Parallel par(maxThreads);

    for (int r = 0; r < repeats; r++) {
        par.doParallelChunked(M, [&](size_t m) {
            double *_a, *_b, *_c;
            __m512d a, b, c;
            for (int n = 0; n < N; n++) {
                _a = &aligned_a[m * K];  // INDEX_ROW(m, 0, M, K)
                                         // = m * K
                _b = &aligned_b[n * K];  // INDEX_COL(0, n, K,
                                         // N) = n * K

                c = _mm512_setzero_pd();
                for (int k = 0; k < K; k += 8) {
                    a = _mm512_load_pd(&_a[k]);
                    b = _mm512_load_pd(&_b[k]);
                    c = _mm512_fmadd_pd(a, b, c);  // c = a * b + c
                }

                _c = (double*)&c;
                aligned_c[INDEX(m, n, M, N)] = _c[0] + _c[1] + _c[2] + _c[3] +
                                               _c[4] + _c[5] + _c[6] + _c[7];
            }
        });
    }
}