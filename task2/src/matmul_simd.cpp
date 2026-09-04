// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {

    for(int i=0 ; i<M ; i++){
        const float *a = A + static_cast<long>(i)* lda;
        for(int j=0 ; j<N ; j++){
            const float *b = B + static_cast<long>(i) * ldb;

            __m256 sum = _mm256_setzero_ps();
            int p=0;
            // Process 8 elements
            for( ; p+7 < K ; p++){
                __m256 va = _mm256_loadu_ps(a + p);

                __m256 vb = _mm256_loadu_ps(b + p);

                sum = _mm256_fmadd_ps(va, vb, sum);
            }

            float temp[8];
            _mm256_storeu_ps(temp, sum);

            float acc = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

            for(; p<K ; p++){
                acc += a[p] * b[p];
            }   

            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}
