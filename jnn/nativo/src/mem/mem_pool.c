#include "mem_pool.h"
#include <stdint.h>
#include <assert.h>
#include <stdlib.h>

_Thread_local mem_pool_t _g_mem_pool = {NULL, 0};

float* get_mem_pool(size_t tam_bytes) {
    if (_g_mem_pool.tam < tam_bytes) {
        if (_g_mem_pool.data != NULL) {
            _aligned_free(_g_mem_pool.data);
        }

        float* novo = _aligned_malloc(tam_bytes, 64);
        assert(novo != NULL && "Ocorreu um erro ao alocar memoria para a pool");

        _g_mem_pool.data = novo;
        _g_mem_pool.tam = tam_bytes;
    }

    return _g_mem_pool.data;
}