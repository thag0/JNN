#include "ocl_context.h"
#include <stdio.h>

int ocl_init(ocl_context_t* ctx) {
    cl_int err;
    cl_uint n;

    ctx->platform = NULL;
    ctx->device   = NULL;
    ctx->context  = NULL;
    ctx->queue    = NULL;

    clGetPlatformIDs(0, NULL, &n);
    cl_platform_id platforms[8];
    clGetPlatformIDs(n, platforms, NULL);

    for (cl_uint i = 0; i < n; i++) {
        cl_uint nd = 0;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &nd);
        if (err == CL_SUCCESS && nd > 0) {
            ctx->platform = platforms[i];
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &ctx->device, NULL);
            if (err == CL_SUCCESS) break;
        }
    }

    if (!ctx->device) {
        printf("ERROR: no OpenCL GPU device found\n");
        return -1;
    }

    ctx->context = clCreateContext(NULL, 1, &ctx->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("ERROR: clCreateContext failed (%d)\n", err);
        return -1;
    }

    ctx->queue = clCreateCommandQueueWithProperties(
        ctx->context, ctx->device, 0, &err
    );
    if (err != CL_SUCCESS) {
        printf("ERROR: clCreateCommandQueue failed (%d)\n", err);
        return -1;
    }

    char name[256];
    clGetDeviceInfo(ctx->device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    printf("OpenCL device: %s\n", name);

    return 0;
}


void ocl_release(ocl_context_t* ctx) {
    clReleaseCommandQueue(ctx->queue);
    clReleaseContext(ctx->context);
}
