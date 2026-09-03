// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

#include <immintrin.h>

#include "matmul.h"
#define TILE 64
void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your best combined implementation.
    // matmul_simd(A, B, C, M, N, K, lda, ldb, ldc);
    for(int i_start = 0; i_start<M ; i_start += TILE){
        for(int j_start = 0; j_start<N; j_start+=TILE){
            int i_end = (i_start+TILE < M)? i_start + TILE: M;
            int j_end = (j_start+TILE < M)? j_start + TILE: M;
            for (int i = i_start; i < i_end; ++i) 
                for (int j = j_start; j < j_end; ++j) 
                    C[i*ldc +j] = 0.0f;
            for(int p_start = 0; p_start<K; p_start += TILE){
                int p_end = (p_start+TILE < M)? p_start + TILE: M;

                for (int i = i_start; i < i_end; ++i) {
                    for (int j = j_start; j < j_end; ++j) {
                        __m512 acc = _mm512_setzero_ps();
                        const float* a = A + static_cast<long>(i) * lda;
                        const float* b = B + static_cast<long>(j) * ldb;
                        int p;
                        for ( p = p_start; p+15 < p_end; p+=16) {
                            if(p+16<K){
                                _mm_prefetch((const char*)&A[i*lda+p+32],_MM_HINT_T0);
                                _mm_prefetch((const char*)&B[j*ldb+p+32],_MM_HINT_T0);

                            }
                            __m512 reg_a = _mm512_loadu_ps(&a[p]);
                            __m512 reg_b = _mm512_loadu_ps(&b[p]);
                            // __m512 reg_c = _mm512_mul_ps(reg_a,reg_b);
                            acc = _mm512_fmadd_ps(reg_a, reg_b, acc);
                            
                        }
                        float horizontal_sum = _mm512_reduce_add_ps(acc);
                        for ( ; p< p_end; p++) {
                            
                            horizontal_sum += a[p]*b[p];
                        }
                        C[static_cast<long>(i) * ldc + j] += horizontal_sum;
                    }
                }
                
                
            }
        }
    }
}
