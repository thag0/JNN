#pragma once

// Transforma os dados de entrada no formato im2col (CHW)
void im2col_3d(
    const float* restrict X,
    float* restrict COL,
    int canais,
    int alt_x, int larg_x,
    int alt_k, int larg_k,
    int alt_pad, int larg_pad,
    int alt_s, int larg_s
);