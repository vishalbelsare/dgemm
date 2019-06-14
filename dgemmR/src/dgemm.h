#ifndef DGEMM_H
#define DGEMM_H

#include <cstdio>   // for printf
#include <cstring>  // used for memset

#define INDEX_COL(i, j, rows, cols) ((i) + ((j) * (rows)))
#define INDEX_ROW(i, j, rows, cols) ((j) + ((i) * (cols)))

#ifdef COLUMN_MAJOR
#define INDEX(i, j, rows, cols) INDEX_COL(i, j, rows, cols)
#else  // ROW_MAJOR
#define INDEX(i, j, rows, cols) INDEX_ROW(i, j, rows, cols)
#endif

extern "C" {
// forward declaration of the blas matrix multiplication since header can
// be missing
// you can look up the declaration probably in /usr/share/R/include/R_ext/BLAS.h
extern void dgemm_(const char* transa,
                   const char* transb,
                   const int* m,
                   const int* n,
                   const int* k,
                   const double* alpha,
                   const double* a,
                   const int* lda,
                   const double* b,
                   const int* ldb,
                   const double* beta,
                   double* c,
                   const int* ldc);
}

namespace dgemm {

void dgemm_C_loops(double* matrix_a,
                   double* matrix_b,
                   double* result,
                   int M,
                   int K,
                   int N,
                   int repeats);

void dgemm_C_blas(double* matrix_a,
                  double* matrix_b,
                  double* result,
                  int M,
                  int K,
                  int N,
                  int repeats);

}  // namespace dgemm

#endif