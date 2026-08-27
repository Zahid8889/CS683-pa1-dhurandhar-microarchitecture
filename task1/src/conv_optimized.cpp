// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {

    const int p = K/2;
    const int in_stride = W+2*p;  

    //Initialize the result with 0s (we will iteratively add in each index)- Ankit
    for(int oy=0 ; oy<H ; oy++){
        for(int ox = 0 ; ox<W ; ox++){
            out[oy*W + ox] = 0.0f;
        }
    }

    //reorder loop: ky->kx->oy->ox - Ankit
    for(int ky=0 ; ky<H ; ky++){
        for(int kx=0 ; kx<W ; kx++){

            float w = ker[ky*K + kx];
            for(int oy=0 ; oy<H ; oy++){
                for(int ox = 0 ; ox<W ; ox++){
                    out[oy*W + ox] += (in[(oy + ky) *in_stride + (ox + kx) ] * w);
                }
            }
        }
    }


}
