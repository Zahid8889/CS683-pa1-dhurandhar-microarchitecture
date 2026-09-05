// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

#define TILE 64

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    
    const int BLOCK = 64;
    const int PREFETCH_DIS = 64;

    for(int ii=0; ii<M ; ii+=BLOCK){
        int i_end = ii+BLOCK<M?ii+BLOCK:M;

        for(int jj=0 ; jj<N ; jj+=BLOCK){
            int j_end = jj+BLOCK<N?jj+BLOCK:N;

            for(int pp=0 ; pp<K ; pp+=BLOCK){

                int p_end = pp+BLOCK<K?pp+BLOCK:K;

                for(int i=ii; i<i_end; i++){
                    const float* a = A+static_cast<long>(i)*lda;

                    for(int j=jj ; j<j_end; j++){
                        const float* b = B+static_cast<long>(j)*ldb;

                        float acc;

                        if(pp==0){
                            acc = 0.0f;
                        }else{
                            acc = C[static_cast<long>(i)*ldc+j];
                        }

                        for(int p=pp ; p<p_end ; p++){

                            if(p+PREFETCH_DIS<p_end){

                                _mm_prefetch(
                                    reinterpret_cast<const char*>(
                                        a+p+PREFETCH_DIS
                                    ),
                                    _MM_HINT_T0
                                );

                                _mm_prefetch(
                                    reinterpret_cast<const char*>(
                                        b+p+PREFETCH_DIS
                                    ),
                                    _MM_HINT_T0
                                );
                            }

                            acc+=a[p]*b[p];
                        }

                        C[static_cast<long>(i)*ldc+j]=acc;

                    }
                }

            }
        }
    }
}