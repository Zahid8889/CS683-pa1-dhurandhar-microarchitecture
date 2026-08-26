// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

// void conv_unroll(const float* in, float* out, const float* ker,
//                  int H, int W, int K) {
//     // TODO(student): replace this placeholder with your unrolled implementation.
//     conv_naive(in, out, ker, H, W, K);
// }
void conv_unroll(const float *in, float *out, const float *ker,
                 int H, int W, int K)
{
    const int p = K / 2;
    const int in_stride = W + 2 * p; // padded row stride

    for (int j = 0; j < H; j++)
    {
        for (int i = 0; i < W; i++)
        {
            out[j * W + i] = 0.0f;
        }
    }

    for (int ky = 0; ky < K; ky++)
    {
        for (int kx = 0; kx < K; kx++)
        {
            float kerv = ker[ky * K + kx];
            for (int oy = 0; oy < H; oy++)
            {
                int ox;
                for (ox = 0; ox + 4 < W; ox += 5)
                {
                    out[oy * W + ox] += in[(oy + ky) * in_stride + (ox + kx)] * kerv;
                    out[oy * W + ox + 1] += in[(oy + ky) * in_stride + (ox + kx + 1)] * kerv;
                    out[oy * W + ox + 2] += in[(oy + ky) * in_stride + (ox + kx + 2)] * kerv;
                    out[oy * W + ox + 3] += in[(oy + ky) * in_stride + (ox + kx + 3)] * kerv;
                    out[oy * W + ox + 4] += in[(oy + ky) * in_stride + (ox + kx + 4)] * kerv;
                }
                for (; ox < W; ox++)
                {
                    out[oy * W + ox] += in[(oy + ky) * in_stride + (ox + kx)] * kerv;
                }
            }
        }
    }
}