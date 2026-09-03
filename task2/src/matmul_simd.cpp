// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.
    // matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            // float acc = 0.0f;
            __m512 acc = _mm512_setzero_ps();
            const float* a = A + static_cast<long>(i) * lda;
            const float* b = B + static_cast<long>(j) * ldb;
            int p;
            for ( p = 0; p+15 < K; p+=16) {
                __m512 reg_a = _mm512_loadu_ps(&a[p]);
                __m512 reg_b = _mm512_loadu_ps(&b[p]);
                // __m512 reg_c = _mm512_mul_ps(reg_a,reg_b);
                acc = _mm512_fmadd_ps(reg_a, reg_b, acc);
                
            }
            float horizontal_sum = _mm512_reduce_add_ps(acc);
            for ( ; p< K; p++) {
                
                horizontal_sum += a[p]*b[p];
            }
            C[static_cast<long>(i) * ldc + j] = horizontal_sum;
        }
    }
}
