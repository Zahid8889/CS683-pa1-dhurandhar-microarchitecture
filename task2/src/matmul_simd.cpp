// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.
    // matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            __m256 va = _mm256_setzero_ps();
            const float* a = A + static_cast<long>(i) * lda;
            const float* b = B + static_cast<long>(j) * ldb;
            for (int p = 0; p < K; p+=8) {
                __m256 vb = _mm256_loadu_ps(a+p);
                __m256 vc = _mm256_loadu_ps(b+p);
                va = _mm256_fmadd_ps(vb,vc,va);
            }
            float temp[8];
            _mm256_storeu_ps(temp, va);

            C[static_cast<long>(i) * ldc + j] = temp[0]+temp[1]+temp[2]+temp[3]+temp[4]+temp[5]+temp[6]+temp[7];
        }
    }
}
