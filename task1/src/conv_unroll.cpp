// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    // TODO(student): replace this placeholder with your unrolled implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for(int oy=0; oy<H; oy++){
        for(int ox=0; ox<W; ox++){
            out[oy*W + ox]=0.0f;
        }
    }

    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            float w = ker[ky * K + kx];
            for (int oy = 0; oy < H; ++oy) {
                for (int ox = 0; ox < W; ox+=4) {
                    out[oy * W + ox] += in[(oy + ky) * in_stride + (ox + kx)] * w;
                    out[oy * W + ox + 1] += in[(oy + ky) * in_stride + (ox + 1 + kx)] * w;
                    out[oy * W + ox + 2] += in[(oy + ky) * in_stride + (ox + 2 + kx)] * w;
                    out[oy * W + ox + 3] += in[(oy + ky) * in_stride + (ox + 3 + kx)] * w;
                }
                // for (; ox < W; ++ox){
                //     out[oy * W + ox] += in[(oy + ky) * in_stride + (ox + kx)] * w;
                // }
            }
        }
    }
    // conv_naive(in, out, ker, H, W, K);
}
