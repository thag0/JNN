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