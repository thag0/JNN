#include <assert.h>
#include "dispatcher.h"

static jnn_backend_t BACKEND_ATUAL = JNN_BACKEND_CPU;

void jnn_set_backend(jnn_backend_t backend) {
    BACKEND_ATUAL = backend;
}
 
void jnn_matmul_dispatcher(const matmul_params_t* params) {
    switch (BACKEND_ATUAL) {
        case JNN_BACKEND_CPU:
            cpu_matmul(params);
        break;

        default:
            cpu_matmul(params);
        break;
    }
}

void jnn_conv2d_fw_dispatcher(const conv2d_fwd_params_t* params) {
    switch (BACKEND_ATUAL) {
        case JNN_BACKEND_CPU:
            cpu_conv2d_forward(params);
        break;
            
        default:
            cpu_conv2d_forward(params);
        break;
    }
}

void jnn_conv2d_bw_dispatcher(const conv2d_bwd_params_t* params) {
    switch (BACKEND_ATUAL) {
        case JNN_BACKEND_CPU:
            cpu_conv2d_backward(params);
        break;
            
        default:
            cpu_conv2d_backward(params);
        break;
    }   
}

void jnn_maxpool2d_fw_dispatcher(const maxpool2d_fwd_params_t* params) {
    switch (BACKEND_ATUAL) {
        case JNN_BACKEND_CPU:
            cpu_maxpool2d_forward(params);
            break;
            
        default:
            cpu_maxpool2d_forward(params);
        break;
    }  
}

void jnn_maxpool2d_bw_dispatcher(const maxpool2d_bwd_params_t* params) {
    switch (BACKEND_ATUAL) {
        case JNN_BACKEND_CPU:
            cpu_maxpool2d_backward(params);
        break;
            
        default:
            cpu_maxpool2d_backward(params);
        break;
    }  
}