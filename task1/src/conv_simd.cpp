// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your AVX2 implementation.
    // conv_naive(in, out, ker, H, W, K);
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    for (int oy = 0; oy < H; ++oy) 
        for (int ox = 0; ox < W; ++ox)
                    out[oy * W + ox]= 0.0f;
                    
    for (int ky = 0; ky < K; ++ky) {
        for (int oy = 0; oy < H; ++oy) {
            for (int kx = 0; kx < K; ++kx) {
                int ox = 0;
                const __m512 w = _mm512_set1_ps(ker[ky * K + kx]);
                for (; ox+15 < W; ox+=16) {
                    __m512 reg_a = _mm512_loadu_ps((in + ((oy + ky) * in_stride + (ox + kx))));
                    __m512 reg_prod = _mm512_mul_ps(reg_a, w);
                    __m512 reg_out = _mm512_loadu_ps(out + (oy * W + ox));
                    __m512 reg_result = _mm512_add_ps(reg_out, reg_prod);
                    _mm512_storeu_ps((out + oy * W + ox), reg_result);
                }
                for(;ox<W;ox++){
                    out[oy * W + ox]+= in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
            }
        }
    }
    
}
