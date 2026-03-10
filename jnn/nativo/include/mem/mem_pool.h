#pragma once

#include <stdlib.h>
#include <stddef.h>
#include <assert.h>

// Workspace de memória
typedef struct mem_pool_t {
    float* data;
    size_t tam;
} mem_pool_t;

// Retorna um bloco de memória do tamanho especificado
// Caso o tamanho requerido seja maior, aloca um novo bloco
float* get_mem_pool(size_t tam_bytes);

// Retorna um bloco de memória para a matriz A do GEMM
float* get_gemm_mem_pool_a(size_t tam);

// Retorna um bloco de memória para a matriz B do GEMM
float* get_gemm_mem_pool_b(size_t tam);