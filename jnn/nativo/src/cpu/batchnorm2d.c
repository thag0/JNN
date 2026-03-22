#include "batchnorm2d.h"
#include <math.h>

void cpu_batchnorm2d_forward(const bn2d_fwd_params_t* params) {
    int lotes   = params->lotes;
    int canais  = params->canais;
    int altX    = params->alt_x;
    int largX   = params->larg_x;

    int areaX = altX * largX;
    int M = lotes * areaX;

    float* x  = params->x;
    float* y  = params->y;
    float* gamma = params->gamma;
    float* beta  = params->beta;
    float* rm = params->media_movel;
    float* rv = params->variancia_movel;

    float* media = params->media;
    float* var   = params->var;
    float* x_norm = params->x_norm;

    float momentum = params->momentum;
    float eps = params->eps;

    if (params->treinando) {
        #pragma omp parallel for
        for (int c = 0; c < canais; c++) {
            double soma = 0.0;

            for (int n = 0; n < lotes; n++) {
                int base = n * canais * areaX + c * areaX;

                for (int i = 0; i < areaX; i++) {
                    soma += x[base + i];
                }
            }

            media[c] = (float)(soma / M);
        }

        #pragma omp parallel for
        for (int c = 0; c < canais; c++) {
            float m = media[c];
            double soma = 0.0;

            for (int n = 0; n < lotes; n++) {
                int base = n * canais * areaX + c * areaX;

                for (int i = 0; i < areaX; i++) {
                    float d = x[base + i] - m;
                    soma += d * d;
                }
            }

            var[c] = (float)(soma / M);
        }

        #pragma omp parallel for
        for (int c = 0; c < canais; c++) {
            float invStd = 1.0f / sqrtf(var[c] + eps);
            float g = gamma[c];
            float b = beta[c];
            float m = media[c];

            for (int n = 0; n < lotes; n++) {
                int base = n * canais * areaX + c * areaX;

                for (int i = 0; i < areaX; i++) {
                    int id = base + i;

                    float norm = (x[id] - m) * invStd;
                    x_norm[id] = norm;
                    y[id] = g * norm + b;
                }
            }

            float var_unbiased = (M > 1) ? (var[c] * M / (M - 1.0f)) : var[c];
            rm[c] = (1.0f - momentum) * rm[c] + momentum * media[c];
            rv[c] = (1.0f - momentum) * rv[c] + momentum * var_unbiased;
        }

    } else {

        #pragma omp parallel for
        for (int c = 0; c < canais; c++) {
            float invStd = 1.0f / sqrtf(rv[c] + eps);
            float g = gamma[c];
            float b = beta[c];
            float m = rm[c];

            for (int n = 0; n < lotes; n++) {
                int base = n * canais * areaX + c * areaX;

                for (int i = 0; i < areaX; i++) {
                    int id = base + i;
                    y[id] = g * (x[id] - m) * invStd + b;
                }
            }
        }
    }  
}

void cpu_batchnorm2d_backward(const bn2d_bwd_params_t* params) {
    const int N = params->lotes;
    const int C = params->canais;
    const int H = params->alt_x;
    const int W = params->larg_x;

    const int area = H * W;
    const int M = N * area;

    float* restrict x_norm = params->x_norm;
    float* restrict var    = params->var;
    float* restrict gamma  = params->gamma;

    float* restrict ge = params->ge;
    float* restrict gs = params->gs;
    float* restrict gg = params->gg;
    float* restrict gb = params->gb;

    for (int c = 0; c < C; c++) {

        double somaGamma = 0.0;
        double somaBeta  = 0.0;

        for (int n = 0; n < N; n++) {
            int base = n*C*area + c*area;

            for (int i = 0; i < area; i++) {
                int id = base + i;
                float grad = gs[id];

                somaGamma += grad * x_norm[id];
                somaBeta  += grad;
            }
        }

        gg[c] += (float)somaGamma;
        gb[c] += (float)somaBeta;
    }

    for (int c = 0; c < C; ++c) {

        float g = gamma[c];
        float inv_std = 1.0f / sqrtf(var[c] + params->eps);

        double somaG = 0.0;
        double somaGENorm = 0.0;

        for (int n = 0; n < N; n++) {
            int base = n*C*area + c*area;

            for (int i = 0; i < area; ++i) {
                int id = base + i;
                float grad = gs[id];

                somaG       += grad;
                somaGENorm  += grad * x_norm[id];
            }
        }

        float coef = g * inv_std / (float)M;

        for (int n = 0; n < N; n++) {
            int base = n*C*area + c*area;

            for (int i = 0; i < area; ++i) {
                int id = base + i;
                float grad  = gs[id];
                float xnorm = x_norm[id];

                ge[id] = coef * ( (float)M * grad - (float)somaG - xnorm * (float)somaGENorm );
            }
        }
    }
}