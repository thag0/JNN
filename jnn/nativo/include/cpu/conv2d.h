#pragma once

#include <stdbool.h>

typedef struct conv2d_fwd_params_t {
    const float* restrict X;
    const float* restrict K;
    const float* restrict B;
    float* restrict DST;

    int lotes;
    int canais;
    int filtros;

    int alt_x;
    int larg_x;
    int alt_k;
    int larg_k;
    int alt_pad;
    int larg_pad;

    bool temBias;
} conv2d_fwd_params_t;

typedef struct {
    const float* restrict X;
    const float* restrict K;
    const float* restrict GS;

    float* restrict GK;
    float* restrict GE;
    float* restrict GB;

    int lotes;
    int canais;
    int filtros;

    int alt_x;
    int larg_x;
    int alt_k;
    int larg_k;
    int alt_pad;
    int larg_pad;

    bool temBias;
} conv2d_bwd_params_t;

// Realiza a propagação direta da camada Conv2D.
void cpu_conv2d_forward(const conv2d_fwd_params_t* params);

// Realiza a propagação reversa da camada Conv2D.
void cpu_conv2d_backward(const conv2d_bwd_params_t* param);
