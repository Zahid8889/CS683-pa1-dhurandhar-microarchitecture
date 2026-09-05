// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your AVX2 implementation.
    // conv_naive(in, out, ker, H, W, K);
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    const int T = 128;

    for(int ty = 0; ty < H; ty+=T){
        for(int tx = 0; tx < W; tx+=T){
            for(int oy=ty; oy < ty + T && oy<H; oy++){
                for(int ox=tx; ox < tx + T && ox<W; ox+=8){
                    __m256 acc = _mm256_setzero_ps();
                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {
                            __m256 va = _mm256_set1_ps(ker[ky * K + kx]);
                            __m256 vb = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx)]);
                            acc = _mm256_fmadd_ps(va,vb,acc);//(va*vb)+vc
                        }
                    }
                    _mm256_storeu_ps(&out[oy * W + ox], acc);
                }
            }
        }
    }
}
