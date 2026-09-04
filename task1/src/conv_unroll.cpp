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

    for (int oy = 0; oy < H; ++oy)
    {
        for (int ox = 0; ox < W; ++ox)
        {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky)
            {
                int kx = 0;
                while (kx + 4 <= K)
                {
                    int out_loc=(oy + ky) * in_stride + (ox + kx);
                    int kernel_loc= ky * K + kx;
                    acc += in[out_loc] * ker[kernel_loc];
                    acc += in[out_loc+ 1] * ker[kernel_loc+ 1];
                    acc += in[out_loc + 2] * ker[kernel_loc + 2];
                    acc += in[out_loc+ 3] * ker[kernel_loc + 3];
                    kx += 4;
                }
                while (kx < K)
                {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                    kx++;
                }
            }
            out[oy * W + ox] = acc;
        }
    }
}