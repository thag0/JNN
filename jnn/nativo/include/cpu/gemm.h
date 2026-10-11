#pragma once

#include <stdbool.h>
#include "arena.h"

// Realiza a multiplicação matricial entre A e B, salvando de DST.
//
// A: Dados de A 
// B: Dados de B 
// C: Dados de C,
// lda: Tamanho da dimensão principal de A
// ldb: Tamanho da dimensão principal de B
// ldc: Tamanho da dimensão principal de C
// M: linhas de A
// K: colunas de A
// N: colunas de B
// trans_a: Flag se A deve ser tratado transposto
// trans_b: Flag se B deve ser tratado transposto
// beta: constante mutiplicativa de C
void cpu_gemm(
    const float* restrict A,
    const float* restrict B,
    float* restrict C,
    int lda,
    int ldb,
    int ldc,
    int M,
    int K,
    int N,
    bool trans_a,
    bool trans_b,
    float beta
);