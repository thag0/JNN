#ifndef JNN_OCL_MATMUL_H
    #define JNN_OCL_MATMUL_H

    #include "ocl_context.h"

    int ocl_matmul(
        ocl_context_t* ctx,
        const float* A,
        const float* B,
        float* C,
        int M, int N, int K
    );

#endif