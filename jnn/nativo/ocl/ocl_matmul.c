#include <stdio.h>
#include <stdlib.h>

#include "ocl_context.h"
#include "ocl_matmul.h"

static cl_program program = NULL;
static cl_kernel kernel   = NULL;

static char* load_kernel(const char* path, size_t* sz) {
    FILE* f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    *sz = ftell(f);
    rewind(f);
    char* src = malloc(*sz + 1);
    fread(src, 1, *sz, f);
    src[*sz] = 0;
    fclose(f);
    return src;
}

static int build_kernel(ocl_context_t* ctx) {
    size_t sz;
    char* src = load_kernel("jnn/nativo/ocl/kernels/matmul.cl", &sz);

    cl_int err;
    program = clCreateProgramWithSource(ctx->context, 1, (const char**)&src, &sz, &err);

    err = clBuildProgram(program, 1, &ctx->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(
            program, ctx->device,
            CL_PROGRAM_BUILD_LOG,
            0, NULL, &log_size
        );

        char* log = malloc(log_size + 1);
        clGetProgramBuildInfo(
            program, ctx->device,
            CL_PROGRAM_BUILD_LOG,
            log_size, log, NULL
        );
        log[log_size] = 0;

        printf("OpenCL build error:\n%s\n", log);
        free(log);
        return -1;
    }

    kernel = clCreateKernel(program, "matmul", &err);

    free(src);
    return err == CL_SUCCESS ? 0 : -1;
}

int ocl_matmul(
    ocl_context_t* ctx,
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    if (!kernel && build_kernel(ctx) != 0) {
        printf("Problema ao buildar o kernel\n");
        return -1;
    }

    cl_int err;
    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    cl_mem dA = clCreateBuffer(
        ctx->context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeA, (void*)A, &err
    );

    cl_mem dB = clCreateBuffer(
        ctx->context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeB, (void*)B, &err
    );

    cl_mem dC = clCreateBuffer(
        ctx->context,
        CL_MEM_WRITE_ONLY,
        sizeC, NULL, &err
    );

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &dC);
    clSetKernelArg(kernel, 3, sizeof(int), &M);
    clSetKernelArg(kernel, 4, sizeof(int), &N);
    clSetKernelArg(kernel, 5, sizeof(int), &K);

    size_t global[2] = { (size_t)M, (size_t)N };

    err = clEnqueueNDRangeKernel(
        ctx->queue,
        kernel,
        2,
        NULL,
        global,
        NULL,
        0, NULL, NULL
    );

    if (err != CL_SUCCESS) {
        printf("ERROR: clEnqueueNDRangeKernel failed (%d)\n", err);
        return -1;
    }

    clFinish(ctx->queue);

    clEnqueueReadBuffer(
        ctx->queue,
        dC,
        CL_TRUE,
        0,
        sizeC,
        C,
        0, NULL, NULL
    );

    clReleaseMemObject(dA);
    clReleaseMemObject(dB);
    clReleaseMemObject(dC);

    return 0;
}