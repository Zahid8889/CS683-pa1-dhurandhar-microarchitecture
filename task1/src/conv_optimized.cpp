// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"

#define TILE 128
void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    // TODO(student): replace this placeholder with your best combined implementation.
    // conv_naive(in, out, ker, H, W, K);
    int p = K/2;
    int in_stride = W + 2*p;

    //  we will start by loop through each block;

    for(int oy1 = 0 ; oy1 < H; oy1+=TILE){
        for(int ox1 = 0 ; ox1 < W; ox1+=TILE){
            //  now for each tile we will work
            int end_oy = (oy1 + TILE <H)?oy1 + TILE:H;
            int end_ox = (ox1 + TILE <W)?ox1 + TILE:W;
            for(int oy = oy1; oy < end_oy; oy++){
                for(int ox = ox1; ox < end_ox; ox++){
                    out[oy*W + ox] = 0.0f;
                }
            }
            for(int ky = 0; ky < K; ky++){
                for(int oy = oy1; oy < end_oy; oy++){
                    for(int kx = 0; kx < K; kx++){
                    // const __m256 w = _mm256_set1_ps(ker[ky * K + kx]);
                        const __m512 w = _mm512_set1_ps(ker[ky * K + kx]);
                        int ox;
                        for(ox = ox1; ox + 15< end_ox; ox+=16){
                            // __m512 reg_a = _mm512_loadu_ps((in + ((oy + ky) * in_stride + (ox + kx))));
                            // __m512 reg_prod = _mm512_mul_ps(reg_a, w);
                            // __m512 reg_out = _mm512_loadu_ps(out + (oy * W + ox));
                            // __m512 reg_result = _mm512_add_ps(reg_out, reg_prod);
                            // _mm512_storeu_ps((out + oy * W + ox), reg_result);

                            __m512 reg_a   = _mm512_loadu_ps(in + ((oy + ky) * in_stride + (ox + kx)));
                            __m512 reg_out = _mm512_loadu_ps(out + (oy * W + ox));
                            reg_out        = _mm512_fmadd_ps(reg_a, w, reg_out);
                            _mm512_storeu_ps(out + (oy * W + ox), reg_out);

                            // __m256 reg_a   = _mm256_loadu_ps(in + ((oy + ky) * in_stride + (ox + kx)));
                            // __m256 reg_out = _mm256_loadu_ps(out + (oy * W + ox));
                            // reg_out        = _mm256_fmadd_ps(reg_a, w, reg_out);
                            // _mm256_storeu_ps(out + (oy * W + ox), reg_out);

                            // __m256 reg_a1   = _mm256_loadu_ps(in + ((oy + ky) * in_stride + (ox + kx)) +8);
                            // __m256 reg_out1 = _mm256_loadu_ps(out + (oy * W + ox) + 8);
                            // reg_out1        = _mm256_fmadd_ps(reg_a1, w, reg_out1);
                            // _mm256_storeu_ps(out + (oy * W + ox) + 8, reg_out1);
                        }
                        for(;ox<end_ox;ox++){
                            out[oy * W + ox]+= in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                        }
                    }
                }
            }
        }
    }
    
}
