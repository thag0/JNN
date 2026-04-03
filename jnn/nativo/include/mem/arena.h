#pragma once

#include <stddef.h>

#define ARENA_ALINHAMENTO 64
#define ARENA_CAP_KB(n) ((size_t) (n) << 10)
#define ARENA_CAP_MB(n) ((size_t) (n) << 20)
#define ARENA_CAP_GB(n) ((size_t) (n) << 30)

// Arena de memória.
typedef struct {
    size_t capacidade;// Capacidade total.
    size_t offset;// Offset atual.
    unsigned char* data;// Dados da arena.
} arena_t;

// Inicializa uma arena de memória com a capacidade especificada.
void arena_init(arena_t* arena, size_t capacidade);

// Aloca um bloco de memória na arena.
// O bloco pode conter um valor um pouco maior para preservar o alinhamento.
// Se o tamanho solicitado for maior que a capacidade da arena, a memória antiga é realocada.
void* arena_alloc(arena_t* arena, size_t size_bytes);

// Reseta o conteúdo da arena.
void arena_reset(arena_t* arena);

// Marca o ponto atual da arena para uma restauração de bloco específica.
size_t arena_checkpoint(arena_t* arena);

// Retorna ao ponto marcado, liberando memória até o ponto marcado.
void arena_restore(arena_t* arena, size_t checkpoint);