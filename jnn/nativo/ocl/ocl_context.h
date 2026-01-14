#ifndef JNN_OCL_CONTEXT_H
    #define JNN_OCL_CONTEXT_H

    #include <CL/cl.h>

    typedef struct {
        cl_platform_id platform;
        cl_device_id   device;
        cl_context     context;
        cl_command_queue queue;
    } ocl_context_t;

    int  ocl_init(ocl_context_t* ctx);
    void ocl_release(ocl_context_t* ctx);

#endif
