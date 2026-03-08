#include "mem_pool.h"
#include <stdint.h>
#include <assert.h>
#include <stdlib.h>

#define _ALINHAMENTO 64

_Thread_local mem_pool_t _g_mem_pool = {NULL, 0};

float* get_mem_pool(size_t tam_bytes) {
    if (_g_mem_pool.tam < tam_bytes) {
        if (_g_mem_pool.data != NULL) {
            _aligned_free(_g_mem_pool.data);
        }

        const size_t bytes = (tam_bytes + 63) & ~63;
        float* novo = _aligned_malloc(bytes, _ALINHAMENTO);
        assert(novo != NULL && "Ocorreu um erro ao alocar memoria para a pool");

        _g_mem_pool.data = novo;
        _g_mem_pool.tam = bytes;
    }

    return _g_mem_pool.data;
}