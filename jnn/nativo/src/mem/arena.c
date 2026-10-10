#include "arena.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static inline size_t alinhar(size_t x, size_t alinhamento) {
    return (x + (alinhamento - 1)) & ~(alinhamento - 1);
}

void arena_init(arena_t* arena, size_t capacidade) {
    arena->capacidade = alinhar(capacidade, ARENA_ALINHAMENTO);
    arena->offset = 0;
    arena->data = _aligned_malloc(arena->capacidade, ARENA_ALINHAMENTO);
    
    if (!arena->data) {
        fprintf(stderr, "RuntimeError: Falha ao alocar dados da arena.\n");
        exit(EXIT_FAILURE);
    }
}

void* arena_alloc(arena_t* arena, size_t size_bytes) {
    size_t offset_alinhado = alinhar(arena->offset, ARENA_ALINHAMENTO);

    if (offset_alinhado > arena->capacidade || size_bytes > arena->capacidade - offset_alinhado) {
        double capacidade = (double)arena->capacidade;
        const char *unidade = "B";

        if (capacidade >= 1024.0 * 1024.0 * 1024.0) {
            capacidade /= 1024.0 * 1024.0 * 1024.0;
            unidade = "GB";
        } else if (capacidade >= 1024.0 * 1024.0) {
            capacidade /= 1024.0 * 1024.0;
            unidade = "MB";
        } else if (capacidade >= 1024.0) {
            capacidade /= 1024.0;
            unidade = "KB";
        }

        fprintf(
            stderr,
            "RuntimeError: Capacidade de %zu bytes (%.2f %s) excedida na arena,",
            arena->capacidade, capacidade, unidade
        );

        fprintf(stderr," considere pre-alocar mais memoria com JNNnative.setTamArena().\n");

        exit(EXIT_FAILURE);
    }

    void* ptr = arena->data + offset_alinhado;
    arena->offset = offset_alinhado + size_bytes;

    return ptr;
}

size_t arena_checkpoint(arena_t* arena) {
    return arena->offset;
}

void arena_restore(arena_t* arena, size_t checkpoint) {
    if (checkpoint > arena->offset) {
        fprintf(stderr, "RuntimeError: Checkpoint inválido para restauração na arena.\n");
        exit(EXIT_FAILURE);
    }

    arena->offset = checkpoint;
}

void arena_reset(arena_t* arena) {
    arena->offset = 0;
}

void arena_free(arena_t* arena) {
    _aligned_free(arena->data);
    arena->data = NULL;
    arena->capacidade = 0;
    arena->offset = 0;
}