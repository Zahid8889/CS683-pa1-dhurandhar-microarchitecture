// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {

    const int p = K/2;
    const int in_stride = W + 2*p;

    const int TILE = 64;

    //initialize
    for(int i=0 ; i<H ; i++){
        for(int j=0 ; j<W ; j++){
            out[i*W+j] = 0.0f;
        }
    }

    for(int tile_y = 0 ; tile_y<H ; tile_y+=TILE){
        for(int tile_x=0 ; tile_x<W ; tile_x+=TILE){
            int x_end = ((tile_x+TILE) < W)?tile_x+TILE:W;
            int y_end = ((tile_y+TILE) < H )?tile_y+TILE:H;

            for(int ky=0 ; ky<K ; ky++){
                for(int kx=0 ; kx<K ; kx++){
                    const float w = ker[ky*K+kx];

                    for(int oy=tile_y ; oy<y_end; oy++){
                        for(int ox=tile_x ; ox<x_end ; ox++){
                            out[oy * W + ox] += in[(oy + ky) * in_stride + (ox + kx)] * w;
                        }
                    }
                }
            }

        }
    }



}
