// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    // TODO(student): replace this placeholder with your unrolled implementation.
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
                for (ox = 0; ox + 3 < W; ox+=4) {
                    // float acc = 0.0f;
                    out[oy * W + ox]    += in[(oy + ky) * in_stride + (ox + kx    )] * ker[ky * K + kx];
                    out[oy * W + ox + 1]+= in[(oy + ky) * in_stride + (ox + kx + 1)] * ker[ky * K + kx];
                    out[oy * W + ox + 2]+= in[(oy + ky) * in_stride + (ox + kx + 2)] * ker[ky * K + kx];
                    out[oy * W + ox + 3]+= in[(oy + ky) * in_stride + (ox + kx + 3)] * ker[ky * K + kx];
                }
                for (; ox< W; ox++)
                    out[oy * W + ox]    += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
            }
            //  = acc;
        }
    }
}
