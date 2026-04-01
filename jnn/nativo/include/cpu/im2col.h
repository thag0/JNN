#pragma once

// Transforma os dados de entrada no formato im2col (CHW).
void im2col(
    const float* restrict X,
    float* restrict COL,
    int canais,
    int alt_x, int larg_x,
    int alt_k, int larg_k,
    int alt_pad, int larg_pad,
    int alt_s, int larg_s
);

//Aplica o col2im transposto
void col2im_T(
    const float* restrict COLT,
    float* restrict GE,
    int canais,
    int alt_x, int larg_x,
    int alt_k, int larg_k,
    int alt_pad, int larg_pad,
    int alt_s, int larg_s
);