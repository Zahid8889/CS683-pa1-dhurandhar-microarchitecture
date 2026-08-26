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
    const int in_stride = W + 2 * p; // padded row stride
    for (int  j = 0; j < H; j++)
    {
        for(int i=0;i<W;i++){
            out[j*W+i]=0.0f;
        }
    }
    
    for (int ky = 0; ky < K; ky++)
    {
        for (int kx = 0; kx < K; kx++)
        {
            float kerv = ker[ky * K + kx];
            __m256 ker = _mm256_set1_ps(kerv);
            for (int oy = 0; oy < H; oy++)
            {
                int ox;
                for ( ox = 0; ox+8<= W; ox+=8)
                {
                    __m256 currPartsum= _mm256_loadu_ps(& in[(oy + ky) * in_stride + (ox + kx)] );
                    __m256 output = _mm256_loadu_ps(&out[oy*W+ox]);
                    output=_mm256_fmadd_ps(currPartsum,ker,output);

                    _mm256_storeu_ps(&out[oy*W+ox],output);
                    
                }
                for(;ox<W;ox++){
                    out[oy*W+ox]+= in[(oy+ky)*in_stride+(ox+kx)]* kerv;
                }
            }
        }
    }
}
