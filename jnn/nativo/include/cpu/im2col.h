#pragma once

// Transforma os dados de entrada no formato im2col (CHW).
void im2col_3d(
    const float* restrict X,
    float* restrict COL,
    int canais,
    int alt_x, int larg_x,
    int alt_k, int larg_k,
    int alt_pad, int larg_pad,
    int alt_s, int larg_s
);

// Versão do im2col que gera a matriz col transposta.
void im2col_3dT(
    const float* restrict X,
    float* restrict COLT,
    int canais,
    int alt_x, int larg_x,
    int alt_k, int larg_k,
    int alt_pad, int larg_pad,
    int alt_s, int larg_s
);

void col2im_3dT(
    const float* restrict COLT,
    float* restrict GE,
    int canais,
    int alt_x, int larg_x,
    int alt_k, int larg_k,
    int alt_pad, int larg_pad,
    int alt_s, int larg_s
);