// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your tiled/blocked implementation.
    // conv_naive(in, out, ker, H, W, K);
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    const int T = 64;

    for(int ty = 0; ty < H; ty+=T){
        for(int tx = 0; tx < W; tx+=T){
            for(int oy=ty; oy < ty + T && oy<H; oy++){
                for(int ox=tx; ox < tx + T && ox<W; ox++){
                    out[oy*W + ox]=0.0f;
                }
            }
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    float w = ker[ky * K + kx];
                    for (int oy = ty; oy < ty + T && oy < H; ++oy) {
                        for (int ox = tx; ox < tx + T && ox < W; ++ox) {
                            out[oy * W + ox] += in[(oy + ky) * in_stride + (ox + kx)] * w;
                        }
                    }
                }
            }
        }
    }
}
