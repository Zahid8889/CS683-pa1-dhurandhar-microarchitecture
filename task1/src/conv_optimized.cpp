// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {

    const int p = K/2;
    const int in_stride = W+2*p;
    
    const int TILE = 64;


    //tiling
    for(int tile_y=0 ; tile_y<H ; tile_y+=TILE){
        for(int tile_x =0 ;tile_x<W ; tile_x+=TILE){

            const int y_end = ((tile_y+TILE) < H)?tile_y+TILE:H;
            const int x_end = ((tile_x + TILE) < W)?tile_x+TILE:W;

            //correct order starts here
            for(int ky=0 ; ky<K ; ky++){
                for(int kx=0 ; kx<K ; kx++){

                    const float w = ker[ky*K+kx];
                    const __m256 weight = _mm256_set1_ps(w);

                    
                    for(int oy=tile_y ; oy<y_end ; oy++){

                        // 4 way unroll inside the loop (few elements may remain)
                        int ox = tile_x;

                        for(; ox+31<x_end ; ox+=32){

                            //apply simd
                            __m256 o0 = _mm256_loadu_ps(&out[oy * W + ox]);
                            __m256 o1 = _mm256_loadu_ps(&out[oy * W + ox + 8]);
                            __m256 o2 = _mm256_loadu_ps(&out[oy * W + ox + 16]);
                            __m256 o3 = _mm256_loadu_ps(&out[oy * W + ox + 24]);

                            __m256 i0 = _mm256_loadu_ps(
                                &in[(oy + ky) * in_stride + ox + kx]);
                            __m256 i1 = _mm256_loadu_ps(
                                &in[(oy + ky) * in_stride + ox + 8 + kx]);
                            __m256 i2 = _mm256_loadu_ps(
                                &in[(oy + ky) * in_stride + ox + 16 + kx]);
                            __m256 i3 = _mm256_loadu_ps(
                                &in[(oy + ky) * in_stride + ox + 24 + kx]);

                            if (ky == 0 && kx == 0) {
                                o0 = _mm256_mul_ps(i0, weight);
                                o1 = _mm256_mul_ps(i1, weight);
                                o2 = _mm256_mul_ps(i2, weight);
                                o3 = _mm256_mul_ps(i3, weight);
                            } else {
                                o0 = _mm256_fmadd_ps(i0, weight, o0);
                                o1 = _mm256_fmadd_ps(i1, weight, o1);
                                o2 = _mm256_fmadd_ps(i2, weight, o2);
                                o3 = _mm256_fmadd_ps(i3, weight, o3);
                            }

                            _mm256_storeu_ps(&out[oy * W + ox], o0);
                            _mm256_storeu_ps(&out[oy * W + ox + 8], o1);
                            _mm256_storeu_ps(&out[oy * W + ox + 16], o2);
                            _mm256_storeu_ps(&out[oy * W + ox + 24], o3);
                        }

                        //handle the remaining elements
                        for(; ox<x_end ; ox+=8){
                            __m256 input = _mm256_loadu_ps(&in[(oy + ky) * in_stride + ox + kx]);

                            __m256 output;

                            if (ky == 0 && kx == 0) {
                                output = _mm256_mul_ps(input, weight);
                            } else {
                                output = _mm256_loadu_ps(&out[oy * W + ox]);
                                output = _mm256_fmadd_ps(input, weight, output);
                            }

                            _mm256_storeu_ps(&out[oy * W + ox], output);
                        }

                    }

                }
            }


        }
    }
    
}
