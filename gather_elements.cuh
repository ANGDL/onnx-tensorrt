//
// Created by ang on 2021/1/11.
//

#ifndef GATHER_ELEMENTS_GATHER_ELEMENTS_CUH
#define GATHER_ELEMENTS_GATHER_ELEMENTS_CUH

#include <cuda.h>
#include <cuda_runtime.h>

void gather_elements(
        const void* const* input,
        void* const* output,
        unsigned int axis,
        int n_dim,
        const int* tensor_dims,
        const int* index_dims,
        void* workspace,
        cudaStream_t stream);

#endif //GATHER_ELEMENTS_GATHER_ELEMENTS_CUH
