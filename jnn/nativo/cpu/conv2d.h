#pragma once
#include <stdbool.h>

typedef struct {
    const double* restrict X;
    const double* restrict K;
    const double* restrict B;
    double* restrict DST;

    int off_x;
    int off_k;
    int off_b;
    int off_dst;

    int lotes;
    int canais;
    int filtros;

    int alt_x;
    int larg_x;
    int alt_k;
    int larg_k;

    bool temBias;
} conv2d_fwd_params_t;

typedef struct {
    const double* restrict X;
    const double* restrict K;
    const double* restrict GS;

    double* restrict GK;
    double* restrict GE;
    double* restrict GB;

    int off_x;
    int off_k;
    int off_gs;
    int off_gk;
    int off_ge;
    int off_gb;

    int lotes;
    int canais;
    int filtros;

    int alt_x;
    int larg_x;
    int alt_k;
    int larg_k;

    bool temBias;
} conv2d_bwd_params_t;

// Realiza a propagação direta da camada Conv2D.
void cpu_conv2d_forward(const conv2d_fwd_params_t* params);

// Realiza a propagação reversa da camada Conv2D.
void cpu_conv2d_backward(const conv2d_bwd_params_t* param);
