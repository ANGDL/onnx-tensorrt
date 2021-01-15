//
// Created by ang on 2021/1/11.
//

#ifndef GATHER_ELEMENTS_GATHER_ELEMENTS_CUH
#define GATHER_ELEMENTS_GATHER_ELEMENTS_CUH

#include <cuda.h>
#include <cuda_runtime.h>

void gather_elements(
        const void* const* input,
        void *const * output,
        unsigned int axis,
        unsigned int t_c, unsigned int t_h, unsigned int t_w,
        unsigned int idx_c, unsigned int idx_h, unsigned int idx_w,
        cudaStream_t stream=0);

#endif //GATHER_ELEMENTS_GATHER_ELEMENTS_CUH
