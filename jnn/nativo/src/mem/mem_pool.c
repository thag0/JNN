#include "mem_pool.h"
#include <stdint.h>
#include <assert.h>
#include <stdlib.h>

#define _ALINHAMENTO 64

_Thread_local mem_pool_t _g_mem_pool      = {NULL, 0};
_Thread_local mem_pool_t _g_gemm_pool_a   = {NULL, 0};
_Thread_local mem_pool_t _g_gemm_pool_b   = {NULL, 0};

// função genérica interna
static float* _get_pool(mem_pool_t* pool, size_t tam_bytes) {
    if (pool->tam < tam_bytes) {
        _aligned_free(pool->data);
        size_t bytes = (tam_bytes + 63) & ~63;
        float* novo  = _aligned_malloc(bytes, _ALINHAMENTO);
        assert(novo != NULL);
        pool->data = novo;
        pool->tam  = bytes;
    }
    return pool->data;
}

float* get_mem_pool(size_t tam)        { return _get_pool(&_g_mem_pool,    tam); }
float* get_gemm_mem_pool_a(size_t tam) { return _get_pool(&_g_gemm_pool_a, tam); }
float* get_gemm_mem_pool_b(size_t tam) { return _get_pool(&_g_gemm_pool_b, tam); }