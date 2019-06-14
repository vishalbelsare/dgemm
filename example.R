rm(list = ls())
library(microbenchmark)
devtools::clean_dll("dgemmR")
devtools::load_all("dgemmR")

set.seed(1)
M <- 40
N <- 500
K <- 52

a <- matrix(rnorm(K * M), nrow = M, ncol = K)
b <- matrix(rnorm(K * N), nrow = K, ncol = N)

c1 <- dgemm_R_blas(a, b)
c2 <- dgemm_R_loops(a, b)
c3 <- dgemm_C_loops(a, b)
c4 <- dgemm_C_blas(a, b)

sum(abs(c1 - c2))
sum(abs(c1 - c3))
sum(abs(c1 - c4))

repeats <- 10
microbenchmark(
    dgemm_R_blas(a, b, repeats),
    dgemm_R_loops(a, b, repeats),
    dgemm_C_loops(a, b, repeats),
    dgemm_C_blas(a, b, repeats),
    unit = "ms", times = 20
)
