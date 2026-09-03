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
                    // for (int j = j_start; j < j_end; ++j) {
                    const float* a = A + static_cast<long>(i) * lda;
                    int j;
                    for ( j = j_start; j +3< j_end; j+=4) {
                        __m512 s0 = _mm512_setzero_ps(),s1= s0, s2 = s1, s3 = s2;
                        // const float* b = B + static_cast<long>(j) * ldb;
                        const float* b0 = B + j * ldb, *b1= b0+ldb, *b2 = b1+ldb, *b3 = b2+ldb;
                        int p;
                        for ( p = p_start; p+15 < p_end; p+=16) {
                            if(p+16<K){
                                // _mm_prefetch((const char*)&A[i*lda+p+32],_MM_HINT_T0);
                                // _mm_prefetch((const char*)&B[j*ldb+p+32],_MM_HINT_T0);
                                _mm_prefetch((const char*)&b0[p+32],_MM_HINT_T0);
                                _mm_prefetch((const char*)&b1[p+32],_MM_HINT_T0);
                                _mm_prefetch((const char*)&b2[p+32],_MM_HINT_T0);
                                _mm_prefetch((const char*)&b3[p+32],_MM_HINT_T0);

                            }
                            __m512 av = _mm512_loadu_ps(a + p);
                            s0 = _mm512_fmadd_ps(av,_mm512_loadu_ps(b0 +p),s0);
                            s1 = _mm512_fmadd_ps(av,_mm512_loadu_ps(b1 +p),s1);
                            s2 = _mm512_fmadd_ps(av,_mm512_loadu_ps(b2 +p),s2);
                            s3 = _mm512_fmadd_ps(av,_mm512_loadu_ps(b3 +p),s3);
                            // __m512 reg_a = _mm512_loadu_ps(&a[p]);
                            // __m512 reg_b = _mm512_loadu_ps(&b[p]);
                            // // __m512 reg_c = _mm512_mul_ps(reg_a,reg_b);
                            // acc = _mm512_fmadd_ps(reg_a, reg_b, acc);
                            
                        }
                        // float horizontal_sum = _mm512_reduce_add_ps(acc);
                        float c0 = _mm512_reduce_add_ps(s0);
                        float c1 = _mm512_reduce_add_ps(s1);
                        float c2 = _mm512_reduce_add_ps(s2);
                        float c3 = _mm512_reduce_add_ps(s3);                        
                        for ( ; p< p_end; p++) {
                            
                            c0 += a[p] * b0[p];
                            c1 += a[p] * b1[p];
                            c2 += a[p] * b2[p];
                            c3 += a[p] * b3[p];
                        }
                        C[static_cast<long>(i) * ldc + j + 0] += c0;
                        C[static_cast<long>(i) * ldc + j + 1] += c1;
                        C[static_cast<long>(i) * ldc + j + 2] += c2;
                        C[static_cast<long>(i) * ldc + j + 3] += c3;
                    }
                    for (; j < j_end; ++j)
                    {
                        const float *b = B + static_cast<long>(j) * ldb;
                        float sum = 0;
                        int p = 0;
                        for (; p + 7 < p_end; p += 8)
                        {
                            
                            __m256 av = _mm256_loadu_ps(a + p), bv = _mm256_loadu_ps(b + p), z = _mm256_mul_ps(av, bv);
                            float t[8];
                            _mm256_storeu_ps(t, z);
                            for (int q = 0; q < 8; ++q)
                                sum += t[q];
                        }
                        for (; p < p_end; ++p)
                            sum += a[p] * b[p];
                        C[static_cast<long>(i) * ldc + j] += sum;
                    }
                }
                
                
            }
        }
    }
}
