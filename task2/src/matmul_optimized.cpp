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

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your best combined implementation.
    // matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
    const int DISTANCE = 64;
    int T = 128;
    for (int bi = 0; bi < M; bi += T) {
        int i_end = (bi + T < M) ? bi + T : M;
        for (int bj = 0; bj < N; bj += T) {
            int j_end = (bj + T < N) ? bj + T : N;

            for (int i = bi; i < i_end; ++i) {
                const float* a = A + (long)i * lda;
                int j = bj;
                for (; j + 3 < j_end; j += 4) {
                    __m256 va1a=_mm256_setzero_ps(), va1b=_mm256_setzero_ps();
                    __m256 va2a=_mm256_setzero_ps(), va2b=_mm256_setzero_ps();
                    __m256 va3a=_mm256_setzero_ps(), va3b=_mm256_setzero_ps();
                    __m256 va4a=_mm256_setzero_ps(), va4b=_mm256_setzero_ps();
                    const float* b1 = B + (long)j*ldb;
                    const float* b2 = B + (long)(j+1)*ldb;
                    const float* b3 = B + (long)(j+2)*ldb;
                    const float* b4 = B + (long)(j+3)*ldb;

                    int p = 0;
                    for (; p + 15 < K; p += 16) {
                        _mm_prefetch((const char*)(a+p+DISTANCE), _MM_HINT_T0);
                        _mm_prefetch((const char*)(b1+p+DISTANCE), _MM_HINT_T0);
                        _mm_prefetch((const char*)(b2+p+DISTANCE), _MM_HINT_T0);
                        _mm_prefetch((const char*)(b3+p+DISTANCE), _MM_HINT_T0);
                        _mm_prefetch((const char*)(b4+p+DISTANCE), _MM_HINT_T0);

                        __m256 va_lo = _mm256_loadu_ps(a+p);
                        __m256 va_hi = _mm256_loadu_ps(a+p+8);

                        va1a = _mm256_fmadd_ps(va_lo, _mm256_loadu_ps(b1+p),   va1a);
                        va1b = _mm256_fmadd_ps(va_hi, _mm256_loadu_ps(b1+p+8), va1b);
                        va2a = _mm256_fmadd_ps(va_lo, _mm256_loadu_ps(b2+p),   va2a);
                        va2b = _mm256_fmadd_ps(va_hi, _mm256_loadu_ps(b2+p+8), va2b);
                        va3a = _mm256_fmadd_ps(va_lo, _mm256_loadu_ps(b3+p),   va3a);
                        va3b = _mm256_fmadd_ps(va_hi, _mm256_loadu_ps(b3+p+8), va3b);
                        va4a = _mm256_fmadd_ps(va_lo, _mm256_loadu_ps(b4+p),   va4a);
                        va4b = _mm256_fmadd_ps(va_hi, _mm256_loadu_ps(b4+p+8), va4b);
                    }
                    for (; p + 7 < K; p += 8) {
                        __m256 vb = _mm256_loadu_ps(a+p);
                        va1a = _mm256_fmadd_ps(vb, _mm256_loadu_ps(b1+p), va1a);
                        va2a = _mm256_fmadd_ps(vb, _mm256_loadu_ps(b2+p), va2a);
                        va3a = _mm256_fmadd_ps(vb, _mm256_loadu_ps(b3+p), va3a);
                        va4a = _mm256_fmadd_ps(vb, _mm256_loadu_ps(b4+p), va4a);
                    }
                    __m256 va1 = _mm256_add_ps(va1a, va1b);
                    __m256 va2 = _mm256_add_ps(va2a, va2b);
                    __m256 va3 = _mm256_add_ps(va3a, va3b);
                    __m256 va4 = _mm256_add_ps(va4a, va4b);
                    float t1[8],t2[8],t3[8],t4[8];
                    _mm256_storeu_ps(t1,va1); _mm256_storeu_ps(t2,va2);
                    _mm256_storeu_ps(t3,va3); _mm256_storeu_ps(t4,va4);
                    float s1=t1[0]+t1[1]+t1[2]+t1[3]+t1[4]+t1[5]+t1[6]+t1[7];
                    float s2=t2[0]+t2[1]+t2[2]+t2[3]+t2[4]+t2[5]+t2[6]+t2[7];
                    float s3=t3[0]+t3[1]+t3[2]+t3[3]+t3[4]+t3[5]+t3[6]+t3[7];
                    float s4=t4[0]+t4[1]+t4[2]+t4[3]+t4[4]+t4[5]+t4[6]+t4[7];
                    for (; p<K; ++p) { s1+=a[p]*b1[p]; s2+=a[p]*b2[p]; s3+=a[p]*b3[p]; s4+=a[p]*b4[p]; }
                    C[(long)i*ldc+j]=s1; C[(long)i*ldc+j+1]=s2;
                    C[(long)i*ldc+j+2]=s3; C[(long)i*ldc+j+3]=s4;
                }
                for (; j < j_end; ++j) {
                    const float* b = B + (long)j*ldb;
                    __m256 va = _mm256_setzero_ps();
                    int p = 0;
                    for (; p+7 < K; p += 8) {
                        _mm_prefetch((const char*)(a+p+DISTANCE), _MM_HINT_T0);
                        _mm_prefetch((const char*)(b+p+DISTANCE), _MM_HINT_T0);
                        va = _mm256_fmadd_ps(_mm256_loadu_ps(a+p), _mm256_loadu_ps(b+p), va);
                    }
                    float t[8]; _mm256_storeu_ps(t, va);
                    float s = t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
                    for (; p<K; ++p) s += a[p]*b[p];
                    C[(long)i*ldc+j] = s;
                }
            }
        }
    }
}
