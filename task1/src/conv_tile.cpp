// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"
#define min(a, b) (((a) < (b)) ? (a) : (b))

void conv_tile(const float *in, float *out, const float *ker,
               int H, int W, int K)
{
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    int TW = 64; 
    int TH = 64; 
    
    for (int ty = 0; ty < H; ty += TH)
    {
        int ty_end = min(ty + TH, H);
        for (int tx = 0; tx < W; tx += TW)
        {
            int tx_end = min(tx + TW, W);

            for (int oy = ty; oy < ty_end; oy++)
            {
                for (int ox = tx; ox + 8 <= tx_end; ox += 8)
                {
                    float acc=0.0f;
                    
                    for (int ky = 0; ky < K; ky++)
                    {
                        for (int kx = 0; kx < K; kx++)
                            acc += in[(oy + ky) * in_stride + ox+kx] * ker[ky*K + kx];
                    }
                    out[oy * W + ox]= acc;
                }
            }
        }
    }
}