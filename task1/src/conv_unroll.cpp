// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    const int p = K/2;
    const int in_stride = W + 2*p;

    //Initialize - Ankit
    for(int i=0 ; i<H ; i++){
        for(int j=0 ; j<W ; j++){
            out[i*K+j] = 0.0f;
        }
    }

    //correct order still
    for(int ky=0 ; ky<K ; ky++){
        for(int kx=0 ; kx<K ; kx++){

            float w = ker[ky*K + kx];

            for(int oy=0 ; oy<H ; oy++){
                for(int ox=0 ; ox<W ; ox+=4){
                    // 4-way unroll
                    out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * w;
                    out[oy*W+ox+1] += in[(oy + ky) * in_stride + (ox +1 + kx)] * w;
                    out[oy*W+ox+2] += in[(oy + ky) * in_stride + (ox +2 + kx)] * w;
                    out[oy*W+ox+3] += in[(oy + ky) * in_stride + (ox +3 + kx)] * w;
                }
            }
        }
    }

}
