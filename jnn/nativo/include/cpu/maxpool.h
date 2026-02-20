#pragma once
#include <float.h>

#define MIN_FLOAT_VAL (-FLT_MAX)

typedef struct {
    const float* restrict X;
    float* restrict Y;

    int lotes;
    int canais;
    int alt_x;
    int larg_x;
    int alt_y;
    int larg_y;
    int alt_pool;
    int larg_pool;
    int alt_std;
    int larg_std;
} maxpool2d_fwd_params_t;

typedef struct {
    const float* restrict X;
    const float* restrict GS;
    float* restrict GE;

    int lotes;
    int canais;
    int alt_x;
    int larg_x;
    int alt_gs;
    int larg_gs;
    int alt_pool;
    int larg_pool;
    int alt_std;
    int larg_std;
} maxpool2d_bwd_params_t;

// Realiza a propagação direta da camada MaxPool2D.
void cpu_maxpool2d_forward(const maxpool2d_fwd_params_t* params);

// Realiza a propagação reversa da camada MaxPool2D.
void cpu_maxpool2d_backward(const maxpool2d_bwd_params_t* params);