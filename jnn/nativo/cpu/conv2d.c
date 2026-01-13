#include "common.h"
#include "conv2d.h"

static inline void corr2d(
    const double* restrict X, int off_x,
    const double* restrict K, int off_k,
    double* restrict DST, int off_dst,
    int alt_x, int larg_x,
    int alt_k, int larg_k
) {
    const int alt_s  = alt_x  - alt_k  + 1;
    const int larg_s = larg_x - larg_k + 1;

    for (int i = 0; i < alt_s; i++) {
        const double* lin_x = X + off_x + i * larg_x;
        double* lin_dst = DST + off_dst + i * larg_s;

        for (int j = 0; j < larg_s; j++) {
            double soma = 0.0;

            for (int kh = 0; kh < alt_k; kh++) {
                const double* px = lin_x + j + kh * larg_x;
                const double* pk = K + off_k + kh * larg_k;

                #pragma omp simd reduction(+:soma)
                for (int kw = 0; kw < larg_k; kw++) {
                    soma += px[kw] * pk[kw];
                }
            }

            lin_dst[j] += soma;
        }
    }

}

static inline void conv2d_full(
    const double* restrict X, int off_x,
    const double* restrict K, int off_k,
    double* restrict DST, int off_dst,
    int alt_x, int larg_x,
    int alt_k, int larg_k
) {
    const int alt_s  = alt_x  + alt_k  - 1;
    const int larg_s = larg_x + larg_k - 1;

    for (int i = 0; i < alt_s; i++) {
        int kh_ini = i >= alt_x ? i - (alt_x - 1) : 0;
        int kh_fim = i < alt_k ? i : alt_k - 1;

        for (int j = 0; j < larg_s; j++) {
            int kw_ini = j >= larg_x ? j - (larg_x - 1) : 0;
            int kw_fim = j < larg_k ? j : larg_k - 1;

            double soma = 0.0;

            for (int kh = kh_ini; kh <= kh_fim; kh++) {
                const int lin_x = i - kh;
                const double* Xp = X + off_x + lin_x * larg_x;
                const double* Kp = K + off_k + kh * larg_k;

                for (int kw = kw_ini; kw <= kw_fim; kw++) {
                    soma += Kp[kw] * Xp[j - kw];
                }
            }

            DST[off_dst + i * larg_s + j] += soma;
        }
    }

}

void cpu_conv2d_forward(const conv2d_fwd_params_t* params) {
    const double* restrict X = params->X;
    const double* restrict K = params->K;
    const double* restrict B = params->B;
    double* restrict DST = params->DST;

    const int alt_x = params->alt_x;
    const int larg_x = params->larg_x;
    const int alt_k = params->alt_k;
    const int larg_k = params->larg_k;
    
    const int lotes = params->lotes;
    const int filtros = params->filtros;
    const int canais = params->canais;

    const int off_x = params->off_x;
    const int off_k = params->off_k;
    const int off_b = params->off_b;
    const int off_dst = params->off_dst;

    bool temBias = params->temBias;

    const int alt_s  = alt_x  - alt_k  + 1;
    const int larg_s = larg_x - larg_k + 1;

    const int area_x = alt_x * larg_x;
    const int area_k = alt_k * larg_k;
    const int area_s = alt_s * larg_s;

    // implementação seca pra enxugar desempenho

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < lotes; b++) {
        for (int f = 0; f < filtros; f++) {
            const int off_x_b  = off_x + b * canais * area_x;
            const int off_y_b  = off_dst + b * filtros * area_s;
            const int off_y_b_f = off_y_b + f * area_s;
            const int off_k_f  = off_k + f * canais * area_k;
            const double bias = temBias ? B[off_b + f] : 0.0;

            for (int i = 0; i < alt_s; i++) {
                double* restrict lin_dst = DST + off_y_b_f + i * larg_s;

                for (int j = 0; j < larg_s; j++) {
                    lin_dst[j] = bias;
                }
            }

            for (int c = 0; c < canais; c++) {
                const int off_x_c = off_x_b + c * area_x;
                const int off_k_c = off_k_f + c * area_k;

                for (int kh = 0; kh < alt_k; kh++) {
                    const int lin_k = off_k_c + kh * larg_k;

                    for (int kw = 0; kw < larg_k; kw++) {
                        const double val_k = K[lin_k + kw];
                        const int x_base = off_x_c + kh * larg_x + kw;

                        for (int i = 0; i < alt_s; i++) {
                            double* restrict lin_dst = DST + off_y_b_f + i * larg_s;
                            const double* restrict lin_x = X + x_base + i * larg_x;

                            #pragma omp simd
                            for (int j = 0; j < larg_s; j++) {
                                lin_dst[j] += lin_x[j] * val_k;
                            }
                        }
                    }
                }
            }
        }
    }
}

void cpu_conv2d_backward(const conv2d_bwd_params_t* params) {
    const double *restrict X  = params->X;
    const double* restrict K  = params->K;
    const double* restrict GS = params->GS;
    double* restrict GK = params->GK;
    double* restrict GE = params->GE;
    double* restrict GB = params->GB;

    bool temBias = params->temBias;

    const int alt_x = params->alt_x;
    const int larg_x = params->larg_x;
    const int alt_k = params->alt_k;
    const int larg_k = params->larg_k;

    const int off_x_base = params->off_x;
    const int off_k_base = params->off_k;
    const int off_gb_base = params->off_gb;
    const int off_gs_base = params->off_gs;
    const int off_ge_base = params->off_ge;
    
    const int lotes = params->lotes;
    const int filtros = params->filtros;
    const int canais = params->canais;

    const int off_x = params->off_x;
    const int off_k = params->off_k;
    const int off_gb = params->off_gb;
    const int off_ge = params->off_ge;

    const int alt_s = alt_x - alt_k + 1;
    const int larg_s = larg_x - larg_k + 1;

    const int area_x = alt_x * larg_x;
    const int area_k = alt_k * larg_k;
    const int area_gs = alt_s * larg_s;

    #pragma omp parallel for schedule(static) 
    for (int f = 0; f < filtros; f++) {
        const int filtro = f;

        const int off_k_f = off_k_base + filtro * canais * area_k;
        double soma_bias_local = 0;

        for (int l = 0; l < lotes; l++) {
            const int off_gs = off_gs_base + (l * filtros + f) * area_gs;
            for (int c = 0; c < canais; c++) {
                int off_x = off_x_base + (l * canais + c) * area_x;
                int off_gk = off_k_f + c * area_k;
                corr2d(
                    X, off_x,
                    GS, off_gs,
                    GK, off_gk,
                    alt_x, larg_x,
                    alt_s, larg_s
                );
            }

            if (temBias) {
                for (int i = 0; i < area_gs; i++) {
                    soma_bias_local += GS[off_gs + i];
                }
            }
        }

        if (temBias) {
            GB[off_gb_base + f] += soma_bias_local;
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int l = 0; l < lotes; l++) {
        for (int c = 0; c < canais; c++) {
            int off_ge = off_ge_base + (l * canais + c) * area_x;
            for (int f = 0; f < filtros; f++) {
                int off_gs = off_gs_base + (l * filtros + f) * area_gs;
                int off_k  = off_k_base  + (f * canais + c) * area_k;
                conv2d_full(
                    GS, off_gs,
                    K,  off_k,
                    GE, off_ge,
                    alt_s, larg_s,
                    alt_k, larg_k
                );
            }
        }
    }
}
