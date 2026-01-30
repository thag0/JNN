#include "conv2d.h"

void cpu_conv2d_forward(const conv2d_fwd_params_t* params) {
    const float* restrict X = params->X;
    const float* restrict K = params->K;
    const float* restrict B = params->B;
    float* restrict DST = params->DST;

    const int alt_x = params->alt_x;
    const int larg_x = params->larg_x;
    const int alt_k = params->alt_k;
    const int larg_k = params->larg_k;
    
    const int lotes = params->lotes;
    const int filtros = params->filtros;
    const int canais = params->canais;

    bool temBias = params->temBias;

    const int alt_s  = alt_x  - alt_k  + 1;
    const int larg_s = larg_x - larg_k + 1;

    const int area_x = alt_x * larg_x;
    const int area_k = alt_k * larg_k;
    const int area_s = alt_s * larg_s;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < lotes; b++) {
        for (int f = 0; f < filtros; f++) {

            const int off_x_b = b * canais * area_x;
            const int off_y   = (b * filtros + f) * area_s;
            const int off_k_f = f * canais * area_k;

            if (temBias) {
                const float bias = B[f];
                for (int i = 0; i < alt_s; i++) {
                    float* restrict dst = DST + off_y + i * larg_s;
                    #pragma omp simd
                    for (int j = 0; j < larg_s; j++) {
                        dst[j] = bias;
                    }
                }
            } else {
                for (int i = 0; i < alt_s; i++) {
                    float* restrict dst = DST + off_y + i * larg_s;
                    #pragma omp simd
                    for (int j = 0; j < larg_s; j++) {
                        dst[j] = 0.0f;
                    }
                }
            }

            for (int c = 0; c < canais; c++) {
                const int off_x_c = off_x_b + c * area_x;
                const int off_k_c = off_k_f + c * area_k;

                for (int kh = 0; kh < alt_k; kh++) {
                    const int lin_k = off_k_c + kh * larg_k;

                    for (int i = 0; i < alt_s; i++) {
                        float* restrict dst = DST + off_y + i * larg_s;
                        const float* restrict lin_X = X + off_x_c + (i + kh) * larg_x;

                        for (int kw = 0; kw < larg_k; kw++) {
                            const float val_k = K[lin_k + kw];
                            const float* restrict x = lin_X + kw;

                            #pragma omp simd
                            for (int j = 0; j < larg_s; j++) {
                                dst[j] += x[j] * val_k;
                            }
                        }
                    }
                }
            }
        }
    }
    
}

void cpu_conv2d_backward(const conv2d_bwd_params_t* params) {
    const float* restrict X  = params->X;
    const float* restrict K  = params->K;
    const float* restrict GS = params->GS;
    float* restrict GK       = params->GK;
    float* restrict GE       = params->GE;
    float* restrict GB       = params->GB;

    const bool temBias = params->temBias;

    const int alt_x = params->alt_x;
    const int larg_x = params->larg_x;
    const int alt_k = params->alt_k;
    const int larg_k = params->larg_k;

    const int lotes  = params->lotes;
    const int filtros = params->filtros;
    const int canais = params->canais;

    const int alt_s  = alt_x  - alt_k  + 1;
    const int larg_s = larg_x - larg_k + 1;

    const int area_x  = alt_x * larg_x;
    const int area_k  = alt_k * larg_k;
    const int area_gs = alt_s * larg_s;

    if (temBias) {
        #pragma omp parallel for schedule(static)
        for (int f = 0; f < filtros; f++) {
            float soma_bias = 0.0f;

            const int f_offset = f * area_gs;

            for (int l = 0; l < lotes; l++) {
                const float* ptr_gs = GS + l * filtros * area_gs + f_offset;

                #pragma omp simd reduction(+:soma_bias)
                for (int i = 0; i < area_gs; i++) {
                    soma_bias += ptr_gs[i];
                }
            }

            GB[f] += soma_bias;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int f = 0; f < filtros; f++) {
        const int f_gs_offset = f * area_gs;
        const int f_k_offset  = f * canais * area_k;

        for (int c = 0; c < canais; c++) {
            float* base_ptr_gk = GK + f_k_offset + c * area_k;

            for (int kh = 0; kh < alt_k; kh++) {
                for (int kw = 0; kw < larg_k; kw++) {
                    float soma = 0.0f;

                    for (int l = 0; l < lotes; l++) {
                        const float* ptr_gs = GS + l * filtros * area_gs + f_gs_offset;
                        const float* ptr_x = X  + l * canais  * area_x  + c * area_x;
                        const float* janela_x = ptr_x + kh * larg_x + kw;

                        for (int i = 0; i < alt_s; i++) {
                            const float* lin_gs = ptr_gs   + i * larg_s;
                            const float* lin_x  = janela_x + i * larg_x;

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

    #pragma omp parallel for collapse(2) schedule(static)
    for (int l = 0; l < lotes; l++) {
        for (int c = 0; c < canais; c++) {
            float* ptr_ge = GE + (l * canais + c) * area_x;

            for (int f = 0; f < filtros; f++) {
                const float* ptr_gs = GS + (l * filtros + f) * area_gs;
                const float* ptr_k  = K  + (f * canais + c) * area_k;

                for (int kh = 0; kh < alt_k; kh++) {
                    for (int kw = 0; kw < larg_k; kw++) {
                        const float val_k = ptr_k[kh * larg_k + kw];
                        float* janela_ge = ptr_ge + kh * larg_x + kw;

                        for (int i = 0; i < alt_s; i++) {
                            const float* lin_gs = ptr_gs   + i * larg_s;
                            float* lin_ge       = janela_ge + i * larg_x;

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
