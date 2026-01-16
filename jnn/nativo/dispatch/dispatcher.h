#pragma once
#include "matmul.h"
#include "conv2d.h"
#include "maxpool.h"

// Tipo de backend nativo.
typedef enum {
    JNN_BACKEND_CPU = 1
} jnn_backend_t;

// Configura um novo backend nativo.
void jnn_set_backend(jnn_backend_t backend);

// Retorna o backend atual de execução
int jnn_get_backend();

// Executa matmul de acordo com o backend nativo atual.
void jnn_matmul_dispatcher(const matmul_params_t* p);

// Executa o forward da camada Conv2D de acordo com o backend nativo atual.
void jnn_conv2d_fw_dispatcher(const conv2d_fwd_params_t* p);

// Executa o backward da camada Conv2D de acordo com o backend nativo atual.
void jnn_conv2d_bw_dispatcher(const conv2d_bwd_params_t* p);

// Executa o forward da camada MaxPool2D de acordo com o backend nativo atual.
void jnn_maxpool2d_fw_dispatcher(const maxpool2d_fwd_params_t* p);

// Executa o backward da camada MaxPool2D de acordo com o backend nativo atual.
void jnn_maxpool2d_bw_dispatcher(const maxpool2d_bwd_params_t* p);