// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "convolution.h"

// void conv_simd(const float* in, float* out, const float* ker,
//                int H, int W, int K) {
//     // TODO(student): replace this placeholder with your AVX2 implementation.
//     conv_simd2(in, out, ker, H, W, K);
// }
void conv_simd(const float *in, float *out, const float *ker,
                 int H, int W, int K)
{
    const int p = K / 2;
    const int in_stride = W + 2 * p; 


    __m256 vec[K*K];
    for(int i=0;i<K*K;i++) vec[i] = _mm256_set1_ps(ker[i]);

    for (int oy = 0; oy < H; ++oy) {
        int ox=0;
        for (ox = 0; ox+8 <= W; ox+=8) {

            __m256 acc = _mm256_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                const float * rownum =  &in[(oy + ky) * in_stride + (ox )];
                for (int kx = 0; kx < K; ++kx) {
                    __m256 v= _mm256_loadu_ps(rownum+ kx );
                    acc= _mm256_fmadd_ps(v,vec[ky*K+kx],acc);
                }
            }
             _mm256_storeu_ps(&out[oy * W + ox], acc);
        }
    }
}
