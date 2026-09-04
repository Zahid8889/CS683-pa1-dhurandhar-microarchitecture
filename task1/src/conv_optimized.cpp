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
    const int T = 64;

    for(int ty = 0; ty < H; ty+=T){
        for(int tx = 0; tx < W; tx+=T){
            for(int oy=ty; oy < ty + T && oy<H; oy++){
                for(int ox=tx; ox < tx + T && ox<W; ox+=32){
                    __m256 zero = _mm256_setzero_ps();
                    _mm256_storeu_ps(&out[oy*W + ox], zero);

                    _mm256_storeu_ps(&out[oy*W + ox + 8], zero);

                    _mm256_storeu_ps(&out[oy*W + ox + 16], zero);

                    _mm256_storeu_ps(&out[oy*W + ox + 24], zero);
                }
            }
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    __m256 va = _mm256_set1_ps(ker[ky * K + kx]);
                    for (int oy = ty; oy < ty + T && oy < H; oy++) {
                        for (int ox = tx; ox < tx + T && ox < W; ox+=32) {
                            __m256 vb = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx)]);
                            __m256 vc = _mm256_loadu_ps(&out[oy * W + ox]);
                            __m256 vr = _mm256_fmadd_ps(va,vb,vc);//(va*vb)+vc
                            _mm256_storeu_ps(&out[oy * W + ox], vr);

                            vb = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + 8 + kx)]);
                            vc = _mm256_loadu_ps(&out[oy * W + ox + 8]);
                            vr = _mm256_fmadd_ps(va,vb,vc);//(va*vb)+vc
                            _mm256_storeu_ps(&out[oy * W + ox +8], vr);

                            vb = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + 16 + kx)]);
                            vc = _mm256_loadu_ps(&out[oy * W + ox + 16]);
                            vr = _mm256_fmadd_ps(va,vb,vc);//(va*vb)+vc
                            _mm256_storeu_ps(&out[oy * W + ox + 16], vr);

                            vb = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + 24 + kx)]);
                            vc = _mm256_loadu_ps(&out[oy * W + ox + 24]);
                            vr = _mm256_fmadd_ps(va,vb,vc);//(va*vb)+vc
                            _mm256_storeu_ps(&out[oy * W + ox + 24], vr);
                        }
                    }
                }
            }
        }
    }
}
