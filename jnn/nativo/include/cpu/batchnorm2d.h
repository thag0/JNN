#pragma once

#include <stdbool.h>

typedef struct {
    float* restrict x;
    float* restrict y;
    float* restrict gamma;
    float* restrict beta;
    float* restrict media_movel;
    float* restrict variancia_movel;
    float* restrict media;
    float* restrict var;
    float* restrict x_norm;
    
    int lotes;
    int canais;
    int alt_x;
    int larg_x;
    
    float momentum;
    float eps;

    bool treinando;
} bn2d_fwd_params_t;

typedef struct {
    float* restrict x_norm;
    float* restrict var;
    float* restrict gamma;
    float* restrict ge;
    float* restrict gs;
    float* restrict gg;
    float* restrict gb;

    int lotes;
    int canais;
    int alt_x;
    int larg_x;
    
    float eps;
} bn2d_bwd_params_t;

// Realiza a propagação direta da camada BatchNorm2D.
void cpu_batchnorm2d_forward(const bn2d_fwd_params_t* params);

// Realiza a propagação reversa da camada BatchNorm2D.
void cpu_batchnorm2d_backward(const bn2d_bwd_params_t* params);