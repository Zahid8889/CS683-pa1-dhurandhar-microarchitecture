// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K/2;
    const int in_stride = W+2*p;

    //Initialize
    for(int oy=0 ; oy<H ; oy++){
        for(int ox=0 ; ox<W ; ox++){
            out[oy*W+ox] = 0.0f;
        }
    }

    for(int ky=0 ; ky<K ; ky++){
        for(int kx=0 ; kx<K ; kx++){

            float w = ker[ky*K + kx];

            __m256 weight = _mm256_set1_ps(w); // a vectos of 8 floats using a single float value
            
            for(int oy=0 ; oy<H ; oy++){
                for(int ox=0 ; ox<W ; ox+=8){
                    
                    __m256 input = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx)]); // loads the input from memory to a vertor of 8 floats
                     
                    __m256 output = _mm256_loadu_ps(&out[oy*W+ox]); 

                    output = _mm256_fmadd_ps(input, weight, output);

                    _mm256_storeu_ps(&out[oy*W+ox], output);

                }
            }

        }
    }


}
