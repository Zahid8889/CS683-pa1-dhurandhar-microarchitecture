// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

void matmul_simd(const float *A, const float *B, float *C,
                 int M, int N, int K, int lda, int ldb, int ldc)
{
    // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.
    // matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
    for (int i = 0; i < M; ++i)
    {
        // for (int j = 0; j < N; ++j) {
        const float *a = A + static_cast<long>(i) * lda;
        int j;
        for ( j = 0; j + 3 < N; j += 4)
        {
            // for (int i = i_start; i < i_end; ++i) 
                for (int jk = j; jk < j+4; ++jk) 
                    C[i*ldc +jk] = 0.0f;
            // float acc = 0.0f;
            // __m256 acc = _mm256_setzero_ps();
            __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s1, s3 = s2;
            // const float* b = B + static_cast<long>(j) * ldb;
            const float *b0 = B + j * ldb, *b1 = b0 + ldb, *b2 = b1 + ldb, *b3 = b2 + ldb;
            int p;
            for (p = 0; p + 7 < K; p += 8)
            {
                // __m256 reg_a = _mm256_loadu_ps(&a[p]);
                // __m256 reg_b = _mm256_loadu_ps(&b[p]);
                // // __m256 reg_c = _mm256_mul_ps(reg_a,reg_b);
                // acc = _mm256_fmadd_ps(reg_a, reg_b, acc);
                __m256 av = _mm256_loadu_ps(a + p);
                s0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b0 + p), s0);
                s1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b1 + p), s1);
                s2 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b2 + p), s2);
                s3 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b3 + p), s3);
            }
            // float horizontal_sum = _mm256_reduce_add_ps(acc);
            // float c0 = _mm256_reduce_add_ps(s0);
            // float c1 = _mm256_reduce_add_ps(s1);
            // float c2 = _mm256_reduce_add_ps(s2);
            // float c3 = _mm256_reduce_add_ps(s3);
            float c0=0, c1=0, c2=0, c3=0;
            float t0[8],t1[8],t2[8],t3[8];
            _mm256_storeu_ps(t0, s0);
            _mm256_storeu_ps(t1, s1);
            _mm256_storeu_ps(t2, s2);
            _mm256_storeu_ps(t3, s3);
            for (int q = 0; q < 8; ++q){
                c0 += t0[q];
                c1 += t1[q];
                c2 += t2[q];
                c3 += t3[q];
            }
            for (; p < K; p++)
            {

                // horizontal_sum += a[p]*b[p];
                c0 += a[p] * b0[p];
                c1 += a[p] * b1[p];
                c2 += a[p] * b2[p];
                c3 += a[p] * b3[p];
            }
            // C[static_cast<long>(i) * ldc + j] = horizontal_sum;
            C[static_cast<long>(i) * ldc + j + 0] += c0;
            C[static_cast<long>(i) * ldc + j + 1] += c1;
            C[static_cast<long>(i) * ldc + j + 2] += c2;
            C[static_cast<long>(i) * ldc + j + 3] += c3;
        }
        for (; j < N; ++j)
        {
            const float *b = B + static_cast<long>(j) * ldb;
            float sum = 0;
            int p = 0;
            for (; p + 7 < K; p += 8)
            {

                __m256 av = _mm256_loadu_ps(a + p), bv = _mm256_loadu_ps(b + p), z = _mm256_mul_ps(av, bv);
                float t[8];
                _mm256_storeu_ps(t, z);
                for (int q = 0; q < 8; ++q)
                    sum += t[q];
            }
            for (; p < K; ++p)
                sum += a[p] * b[p];
            C[static_cast<long>(i) * ldc + j] += sum;
        }
    }
}
