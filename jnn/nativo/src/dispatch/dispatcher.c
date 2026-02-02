#include <stdio.h>
#include "dispatcher.h"

static jnn_backend_t BACKEND_ATUAL = JNN_BACKEND_CPU;

void jnn_set_backend(jnn_backend_t backend) {
    BACKEND_ATUAL = backend;
}

int jnn_get_backend() {
    return BACKEND_ATUAL;
}
 
void jnn_matmul_dispatcher(const matmul_params_t* p) {
    switch (BACKEND_ATUAL) {
        case JNN_BACKEND_CPU:
            cpu_matmul(p);
        break;

        default: cpu_matmul(p);
    }
}

void jnn_conv2d_fw_dispatcher(const conv2d_fwd_params_t* p) {
    switch (BACKEND_ATUAL) {
        case JNN_BACKEND_CPU:
            cpu_conv2d_forward(p);
        break;
            
        default: cpu_conv2d_forward(p);
    }
}

void jnn_conv2d_bw_dispatcher(const conv2d_bwd_params_t* p) {
    switch (BACKEND_ATUAL) {
        case JNN_BACKEND_CPU:
            cpu_conv2d_backward(p);
        break;
            
        default: cpu_conv2d_backward(p);
    }   
}

void jnn_maxpool2d_fw_dispatcher(const maxpool2d_fwd_params_t* p) {
    switch (BACKEND_ATUAL) {
        case JNN_BACKEND_CPU:
            cpu_maxpool2d_forward(p);
        break;
            
        default: cpu_maxpool2d_forward(p);
    }  
}

void jnn_maxpool2d_bw_dispatcher(const maxpool2d_bwd_params_t* p) {
    switch (BACKEND_ATUAL) {
        case JNN_BACKEND_CPU:
            cpu_maxpool2d_backward(p);
        break;
            
        default: cpu_maxpool2d_backward(p);
    }  
}