// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"
#define TILE 32
void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your tiled/blocked implementation.
    // conv_naive(in, out, ker, H, W, K);
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    for (int oy = 0; oy < H; ++oy) 
        for (int ox = 0; ox < W; ++ox)
            out[oy * W + ox]= 0.0f;

    for (int oy = 0; oy < H; oy+=TILE) {
        for (int ox = 0; ox < W; ox+=TILE) {
            int endx = (ox+TILE <H)?ox+TILE:H;
            int endy = (oy+TILE <W)?oy+TILE:W;
            // float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int ny = oy; ny < endy; ++ny) {
                        for (int kx = 0; kx < K; ++kx) {
                        for (int nx = ox; nx < endx; ++nx) {
                            // float acc = 0.0f;
                            out[ny * W + nx]+= in[(ny + ky) * in_stride + (nx + kx)] * ker[ky * K + kx];
                        }
                    }
                }
            }
        }
    }
}
