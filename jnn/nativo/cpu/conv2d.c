#include "conv2d.h"

// isso aqui ta bom, mas ta dificil de ler já

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

    int BI = 4;
    int BJ = 16;

    #pragma omp parallel for schedule(static)
    for (int bf = 0; bf < lotes * filtros; bf++) {
        int b = bf / filtros;
        int f = bf % filtros;
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

void cpu_conv2d_backward(const conv2d_bwd_params_t* params) {
    const double* restrict X = params->X;
    const double* restrict K = params->K;
    const double* restrict GS = params->GS;
    double* restrict GK = params->GK;
    double* restrict GE = params->GE;
    double* restrict GB = params->GB;

    bool temBias = params->temBias;

    const int alt_x = params->alt_x;
    const int larg_x = params->larg_x;
    const int alt_k = params->alt_k;
    const int larg_k = params->larg_k;
    
    const int lotes = params->lotes;
    const int filtros = params->filtros;
    const int canais = params->canais;

    const int off_x_base = params->off_x;
    const int off_k_base = params->off_k;
    const int off_gb_base = params->off_gb;
    const int off_gs_base = params->off_gs;
    const int off_ge_base = params->off_ge;

    const int alt_s = alt_x - alt_k + 1;
    const int larg_s = larg_x - larg_k + 1;

    const int area_x = alt_x * larg_x;
    const int area_k = alt_k * larg_k;
    const int area_gs = alt_s * larg_s;

    #pragma omp parallel for schedule(static)
    for (int f = 0; f < filtros; f++) {
        if (temBias) {
            double soma_bias = 0.0;
            for (int l = 0; l < lotes; l++) {
                const double* ptr_gs = GS + off_gs_base + (l * filtros + f) * area_gs;
                #pragma omp simd reduction(+:soma_bias)
                for (int i = 0; i < area_gs; i++) {
                    soma_bias += ptr_gs[i];
                }
            }
            GB[off_gb_base + f] += soma_bias;
        }

        // grad kernel
        for (int c = 0; c < canais; c++) {
            double* base_ptr_gk = GK + off_k_base + (f * canais + c) * area_k;

            for (int kh = 0; kh < alt_k; kh++) {
                for (int kw = 0; kw < larg_k; kw++) {
                    double soma = 0.0;
                    
                    for (int l = 0; l < lotes; l++) {
                        const double* ptr_gs = GS + off_gs_base + (l * filtros + f) * area_gs;
                        const double* ptr_x  = X + off_x_base + (l * canais + c) * area_x;
                        const double* janela_x = ptr_x + kh * larg_x + kw;
                        
                        for (int i = 0; i < alt_s; i++) {
                            const double* lin_gs = ptr_gs + i * larg_s;
                            const double* lin_x  = janela_x + i * larg_x;

                            #pragma omp simd reduction(+:soma)
                            for (int j = 0; j < larg_s; j++) {
                                soma += lin_gs[j] * lin_x[j];
                            }
                        }
                    }

                    base_ptr_gk[kh * larg_k + kw] += soma;
                }
            }
        }
    }

    // grad entrada
    #pragma omp parallel for collapse(2) schedule(static)
    for (int l = 0; l < lotes; l++) {
        for (int c = 0; c < canais; c++) {
            double* ptr_ge = GE + off_ge_base + (l * canais + c) * area_x;

            for (int f = 0; f < filtros; f++) {
                const double* ptr_gs = GS + off_gs_base + (l * filtros + f) * area_gs;
                const double* ptr_k  = K + off_k_base + (f * canais + c) * area_k;

                for (int kh = 0; kh < alt_k; kh++) {
                    for (int kw = 0; kw < larg_k; kw++) {
                        const double val_k = ptr_k[kh * larg_k + kw];
                        if (val_k == 0.0) continue;
                        
                        double* janela_ge = ptr_ge + kh * larg_x + kw;

                        for (int i = 0; i < alt_s; i++) {
                            const double* lin_gs = ptr_gs + i * larg_s;
                            double* lin_ge = janela_ge + i * larg_x;

                            #pragma omp simd
                            for (int j = 0; j < larg_s; j++) {
                                lin_ge[j] += lin_gs[j] * val_k;
                            }
                        }
                    }
                }
            }
        }
    }
}
