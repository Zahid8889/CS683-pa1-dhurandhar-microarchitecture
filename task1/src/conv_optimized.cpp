// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    // TODO(student): replace this placeholder with your best combined implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    const int T = 128;

    for(int ty = 0; ty < H; ty+=T){
        for(int tx = 0; tx < W; tx+=T){
            for(int oy=ty; oy < ty + T && oy<H; oy++){
                const float* in_row_base = in + (long)oy * in_stride;
                for(int ox=tx; ox < tx + T && ox<W; ox+=32){
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();

                    for (int ky = 0; ky < K; ++ky) {
                        const float* row = in_row_base + (long)ky * in_stride;
                        for (int kx = 0; kx < K; ++kx) {
                            __m256 va = _mm256_set1_ps(ker[ky * K + kx]);
 
                            __m256 vb0 = _mm256_loadu_ps(row + ox + kx);
                            __m256 vb1 = _mm256_loadu_ps(row + ox + 8 + kx);
                            __m256 vb2 = _mm256_loadu_ps(row + ox + 16 + kx);
                            __m256 vb3 = _mm256_loadu_ps(row + ox + 24 + kx);
 
                            acc0 = _mm256_fmadd_ps(va, vb0, acc0);
                            acc1 = _mm256_fmadd_ps(va, vb1, acc1);
                            acc2 = _mm256_fmadd_ps(va, vb2, acc2);
                            acc3 = _mm256_fmadd_ps(va, vb3, acc3);
                        }
                    }
                    _mm256_storeu_ps(&out[(long)oy * W + ox], acc0);
                    _mm256_storeu_ps(&out[(long)oy * W + ox + 8], acc1);
                    _mm256_storeu_ps(&out[(long)oy * W + ox + 16], acc2);
                    _mm256_storeu_ps(&out[(long)oy * W + ox + 24], acc3);
                }
            }
        }
    }
}

