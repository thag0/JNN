#pragma once

#include <stdlib.h>
#include <stddef.h>
#include <assert.h>

// Workspace de memória
typedef struct {
    size_t tam;
    void* data;
} mem_pool_t;

// Retorna um bloco de memória do tamanho especificado
// Caso o tamanho requerido seja maior, aloca um novo bloco
void* get_mem_pool(size_t tam_bytes);

// Retorna um bloco de memória para a matriz A da GEMM
void* get_gemm_mem_pool_a(size_t tam_bytes);

// Retorna um bloco de memória para a matriz B da GEMM
void* get_gemm_mem_pool_b(size_t tam_bytes);