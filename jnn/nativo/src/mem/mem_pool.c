#include "mem_pool.h"
#include <stdint.h>
#include <assert.h>
#include <stdlib.h>

#define _ALINHAMENTO 64

_Thread_local mem_pool_t _g_mem_pool    = {0, NULL};
_Thread_local mem_pool_t _g_gemm_pool_a = {0, NULL};
_Thread_local mem_pool_t _g_gemm_pool_b = {0, NULL};

static void* _get_pool(mem_pool_t* pool, size_t tam_bytes) {
    if (pool->tam < tam_bytes) {
        if (pool->data != NULL) _aligned_free(pool->data);
        
        size_t bytes = (tam_bytes + 63) & ~63;
        void* novo  = _aligned_malloc(bytes, _ALINHAMENTO);
        assert(novo != NULL && "Falha ao alocar memória para o pool");

        pool->data = novo;
        pool->tam  = bytes;
    }

    return pool->data;
}

void* get_mem_pool(size_t tam_bytes) { 
    return _get_pool(&_g_mem_pool, tam_bytes);
}

void* get_gemm_mem_pool_a(size_t tam_bytes) {
    return _get_pool(&_g_gemm_pool_a, tam_bytes);
}

void* get_gemm_mem_pool_b(size_t tam_bytes) {
    return _get_pool(&_g_gemm_pool_b, tam_bytes);
}