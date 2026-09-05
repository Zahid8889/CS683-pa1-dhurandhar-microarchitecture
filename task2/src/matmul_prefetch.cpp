// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your cache-blocked SIMD + prefetch
    // implementation.
    // matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
    const int DISTANCE = 32;
    int T = 32;
    for(int bi = 0; bi < M; bi+=T){
        for(int bj = 0; bj < N; bj+=T){
            for(int i=bi; i < bi+T && i<M; i++){
                for(int j=bj; j < bj+T && j<N; j++){
                    C[static_cast<long>(i) * ldc + j]=0.0f;
                }
            }

            for(int bk = 0; bk < K; bk+=T){
                for (int i = bi; i < bi + T && i < M; ++i) {
                    for (int j = bj; j < bj + T && j < N; ++j) {
                        float acc = C[static_cast<long>(i) * ldc + j];
                        const float* a = A + static_cast<long>(i) * lda;
                        const float* b = B + static_cast<long>(j) * ldb;
                        for (int p = bk; p < bk + T && p < K; ++p) {
                            _mm_prefetch((const char*)(a+p+DISTANCE), _MM_HINT_T0);
                            _mm_prefetch((const char*)(b+p+DISTANCE), _MM_HINT_T0);
                            acc += a[p] * b[p];
                        }
                        C[static_cast<long>(i) * ldc + j] = acc;
                    }
                }
            }
        }
    }
}
