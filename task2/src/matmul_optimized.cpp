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

static inline float hsum256(__m256 x) {
    float temp[8];
    _mm256_storeu_ps(temp, x);

    return temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
}

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K,
                      int lda, int ldb, int ldc) {

    const int BLOCK = 8;
    const int PREFETCH_DISTANCE = 8;

    // Cache blocking 
    for (int ii = 0; ii < M; ii += BLOCK) {

        const int i_end = (ii + BLOCK < M) ? ii + BLOCK : M;

        for (int jj = 0; jj < N; jj += BLOCK) {

            const int j_end = (jj + BLOCK < N) ? jj + BLOCK : N;

            // Inside block now
            for (int i = ii; i < i_end; ++i) {

                const float* a = A + static_cast<long>(i) * lda;

                int j = jj;
                // 4-way register blocking
                for (; j + 3 < j_end; j += 4) {
                    const float* b0 = B + static_cast<long>(j) * ldb;
                    const float* b1 = B + static_cast<long>(j + 1) * ldb;
                    const float* b2 = B + static_cast<long>(j + 2) * ldb;
                    const float* b3 =  B + static_cast<long>(j + 3) * ldb;

                    __m256 s0 = _mm256_setzero_ps();
                    __m256 s1 = _mm256_setzero_ps();
                    __m256 s2 = _mm256_setzero_ps();
                    __m256 s3 = _mm256_setzero_ps();

                    int p = 0;

                    // SIMD and 2-way unroll
                    for (; p + 15 < K; p += 16) {
                        // Software prefetch every cache-line boundary.
                        if (p + PREFETCH_DISTANCE < K) {
                            _mm_prefetch(reinterpret_cast<const char*>( a + p + PREFETCH_DISTANCE), _MM_HINT_T0);
                            _mm_prefetch(reinterpret_cast<const char*>( b0 + p + PREFETCH_DISTANCE), _MM_HINT_T0);
                            _mm_prefetch( reinterpret_cast<const char*>( b1 + p + PREFETCH_DISTANCE), _MM_HINT_T0);
                            _mm_prefetch( reinterpret_cast<const char*>( b2 + p + PREFETCH_DISTANCE), _MM_HINT_T0);
                            _mm_prefetch( reinterpret_cast<const char*>( b3 + p + PREFETCH_DISTANCE), _MM_HINT_T0);
                        }

                        // First 8 floats
                        __m256 va0 = _mm256_loadu_ps(a + p);
                        __m256 vb00 = _mm256_loadu_ps(b0 + p);
                        __m256 vb10 = _mm256_loadu_ps(b1 + p);
                        __m256 vb20 = _mm256_loadu_ps(b2 + p);
                        __m256 vb30 = _mm256_loadu_ps(b3 + p);

                        s0 = _mm256_fmadd_ps(va0, vb00, s0);
                        s1 = _mm256_fmadd_ps(va0, vb10, s1);
                        s2 = _mm256_fmadd_ps(va0, vb20, s2);
                        s3 = _mm256_fmadd_ps(va0, vb30, s3);

                        // Next 8 floats
                        __m256 va1 = _mm256_loadu_ps(a + p + 8);
                        __m256 vb01 = _mm256_loadu_ps(b0 + p + 8);
                        __m256 vb11 = _mm256_loadu_ps(b1 + p + 8);
                        __m256 vb21 = _mm256_loadu_ps(b2 + p + 8);
                        __m256 vb31 = _mm256_loadu_ps(b3 + p + 8);

                        s0 = _mm256_fmadd_ps(va1, vb01, s0);
                        s1 = _mm256_fmadd_ps(va1, vb11, s1);
                        s2 = _mm256_fmadd_ps(va1, vb21, s2);
                        s3 = _mm256_fmadd_ps(va1, vb31, s3);
                    }

                    // Remaining complete SIMD chunk
                    for (; p + 7 < K; p += 8) {
                        if (p + PREFETCH_DISTANCE < K) {
                            _mm_prefetch( reinterpret_cast<const char*>( a + p + PREFETCH_DISTANCE), _MM_HINT_T0);
                            _mm_prefetch( reinterpret_cast<const char*>( b0 + p + PREFETCH_DISTANCE),_MM_HINT_T0);
                            _mm_prefetch(reinterpret_cast<const char*>(b1 + p + PREFETCH_DISTANCE),_MM_HINT_T0);
                            _mm_prefetch(reinterpret_cast<const char*>(b2 + p + PREFETCH_DISTANCE),_MM_HINT_T0);
                            _mm_prefetch(reinterpret_cast<const char*>(b3 + p + PREFETCH_DISTANCE),_MM_HINT_T0);
                        }
                        __m256 va = _mm256_loadu_ps(a + p);
                        s0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b0 + p), s0);
                        s1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b1 + p), s1);
                        s2 = _mm256_fmadd_ps( va, _mm256_loadu_ps(b2 + p), s2);
                        s3 = _mm256_fmadd_ps(va,_mm256_loadu_ps(b3 + p),s3);
                    }

                    float r0 = hsum256(s0);
                    float r1 = hsum256(s1);
                    float r2 = hsum256(s2);
                    float r3 = hsum256(s3);

                    // final leftovers
                    for (; p < K; ++p) {
                        r0 += a[p] * b0[p];
                        r1 += a[p] * b1[p];
                        r2 += a[p] * b2[p];
                        r3 += a[p] * b3[p];
                    }

                    C[static_cast<long>(i) * ldc + j]     = r0;
                    C[static_cast<long>(i) * ldc + j + 1] = r1;
                    C[static_cast<long>(i) * ldc + j + 2] = r2;
                    C[static_cast<long>(i) * ldc + j + 3] = r3;
                }

                // Remaining columns
                for (; j < j_end; ++j) {

                    const float* b =B + static_cast<long>(j) * ldb;

                    __m256 sum = _mm256_setzero_ps();

                    int p = 0;

                    for (; p + 7 < K; p += 8) {
                        if (p + PREFETCH_DISTANCE < K) {
                            _mm_prefetch(reinterpret_cast<const char*>(a + p + PREFETCH_DISTANCE),_MM_HINT_T0);
                            _mm_prefetch(reinterpret_cast<const char*>(b + p + PREFETCH_DISTANCE),_MM_HINT_T0);
                        }
                        __m256 va = _mm256_loadu_ps(a + p);
                        __m256 vb = _mm256_loadu_ps(b + p);
                        sum = _mm256_fmadd_ps(va, vb, sum);
                    }

                    float result = hsum256(sum);

                    for (; p < K; ++p) {
                        result += a[p] * b[p];
                    }

                    C[static_cast<long>(i) * ldc + j] = result;
                }
            }
        }
    }
}