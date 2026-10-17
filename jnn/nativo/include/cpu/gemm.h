#pragma once

#include <stdbool.h>
#include "arena.h"

/* Realiza a multiplicação matricial entre A e B, acumulando o resutado em C.
 *
 * trans_a: Flag se A deve ser tratado transposto.
 * trans_b: Flag se B deve ser tratado transposto.
 * M: Linhas de A.
 * N: Colunas de B.
 * K: Colunas de A.
 * A: Dados de A.
 * lda: Tamanho da dimensão principal de A.
 * B: Dados de B.
 * ldb: Tamanho da dimensão principal de B.
 * beta: Constante mutiplicativa de C.
 * C: Dados de C.
 * ldc: Tamanho da dimensão principal de C.
 */
void cpu_gemm(
    bool trans_a,
    bool trans_b,

    int M,
    int N,
    int K,

    const float* restrict A,
    int lda,
    const float* restrict B,
    int ldb,
    
    float beta,
    float* restrict C,
    int ldc
);