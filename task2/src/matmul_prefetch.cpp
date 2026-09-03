// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"
#define TILE 64
void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your cache-blocked SIMD + prefetch
    // implementation.
    // matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
    
    for(int i1 = 0 ; i1<M; i1+=TILE){
        for(int j1 = 0 ; j1<N; j1+=TILE){
            int endi = (i1+TILE<M)? i1+TILE:M;
            int endj = (j1+TILE<N)? j1+TILE:N;
            
            for (int i = i1; i < endi; ++i) 
                for (int j = j1; j < endj; ++j) 
                    C[i*ldc +j] = 0.0f;

            for(int k1 = 0 ; k1<K; k1+=TILE){
                
                int endk = (k1+TILE<K)? k1+TILE:K;

                for (int i = i1; i < endi; ++i) {
                    for (int j = j1; j < endj; ++j) {
                        float acc = 0.0f;
                        const float* a = A + static_cast<long>(i) * lda;
                        const float* b = B + static_cast<long>(j) * ldb;
                        for (int p = k1; p < endk; ++p) {
                            if(p+16<endk){
                                _mm_prefetch((const char*)&A[i*lda+p+32],_MM_HINT_T0);
                                _mm_prefetch((const char*)&B[j*ldb+p+32],_MM_HINT_T0);

                            }
                            acc += a[p] * b[p];
                        }
                        C[static_cast<long>(i) * ldc + j] += acc;
                    }
                }
            }
        }
    }
    
}
