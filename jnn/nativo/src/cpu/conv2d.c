#include "conv2d.h"
#include "macros.h"
#include "matmul.h"
#include "im2col.h"

#include <stdlib.h>

// forward

static void _forward_loops(const conv2d_fwd_params_t* params) {
    const float* restrict X = params->X;
    const float* restrict K = params->K;
    const float* restrict B = params->B;
    float* restrict DST = params->DST;

    const int alt_x = params->alt_x;
    const int larg_x = params->larg_x;
    const int alt_k = params->alt_k;
    const int larg_k = params->larg_k;
    const int alt_pad = params->alt_pad;
    const int larg_pad = params->larg_pad;
    
    const int lotes = params->lotes;
    const int filtros = params->filtros;
    const int canais = params->canais;

    const bool temBias = params->temBias;

    const int alt_s  = (alt_x  + 2 * alt_pad  - alt_k ) + 1;
    const int larg_s = (larg_x + 2 * larg_pad - larg_k) + 1;

    const int area_x = alt_x * larg_x;
    const int area_k = alt_k * larg_k;
    const int area_s = alt_s * larg_s;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int l = 0; l < lotes; l++) {
        for (int f = 0; f < filtros; f++) {
            float* restrict dst_base = DST + (l * filtros + f) * area_s;
            for (int i = 0; i < area_s; i++) dst_base[i] = temBias ? B[f] : 0.0f;

            for (int c = 0; c < canais; c++) {
                const float* restrict Xc = X + (l * canais + c) * area_x;
                const float* restrict Kc = K + (f * canais + c) * area_k;

                for (int kh = 0; kh < alt_k; kh++) {
                    for (int kw = 0; kw < larg_k; kw++) {
                        const float val_k = Kc[kh * larg_k + kw];
                        const int j_max = MIN_ENTRE(larg_x + larg_pad - kw, larg_s);
                        const int j_min = MAX_ENTRE(larg_pad - kw, 0);
                        const int i_max = MIN_ENTRE(alt_x + alt_pad - kh, alt_s);
                        const int i_min = MAX_ENTRE(alt_pad - kh, 0);

                        for (int i = i_min; i < i_max; i++) {
                            const int in_y = i + kh - alt_pad;
                            float* restrict ptr_dst = dst_base + i * larg_s;
                            const float* restrict ptr_x = Xc + in_y * larg_x;

                            #pragma omp simd
                            for (int j = j_min; j < j_max; j++) {
                                int in_x = j + kw - larg_pad;
                                ptr_dst[j] += ptr_x[in_x] * val_k;
                            }
                        }
                    }
                }
            }
        }
    }
}

static void _forward_im2col(const conv2d_fwd_params_t* params) {
    const float* restrict X = params->X;
    const float* restrict K = params->K;
    const float* restrict B = params->B;
    float* restrict DST = params->DST;

    const int lotes   = params->lotes;
    const int filtros = params->filtros;
    const int canais  = params->canais;

    const int alt_x = params->alt_x;
    const int larg_x = params->larg_x;
    const int alt_k = params->alt_k;
    const int larg_k = params->larg_k;
    const int alt_pad = params->alt_pad;
    const int larg_pad = params->larg_pad;

    const bool temBias = params->temBias;

    const int alt_s  = alt_x  + 2 * alt_pad  - alt_k  + 1;
    const int larg_s = larg_x + 2 * larg_pad - larg_k + 1;

    const int Kdim = canais * alt_k * larg_k;
    const int Ndim = alt_s * larg_s;

    float* col = malloc(sizeof(float) * Kdim * Ndim);

    matmul_params_t mm = {
        .A = (float*) K,
        .B = col,
        .DST = NULL,

        .off_a = 0,
        .off_b = 0,
        .off_dst = 0,

        .std_a_0 = Kdim,
        .std_a_1 = 1,

        .std_b_0 = Ndim,
        .std_b_1 = 1,

        .std_c_0 = Ndim,
        .std_c_1 = 1,

        .lin_a = filtros,
        .col_a = Kdim,
        .col_b = Ndim
    };

    for (int l = 0; l < lotes; l++) {
        const float* x_lote = X + l * canais * alt_x * larg_x;
        float* y_lote = DST + l * filtros * Ndim;

        for (int f = 0; f < filtros; f++) {
            float bias = temBias ? B[f] : 0.f;
            float* linha = y_lote + f * Ndim;
            for (int i = 0; i < Ndim; i++) {
                linha[i] = bias;
            }
        }

        im2col_3d(
            x_lote,
            col,
            canais,
            alt_x, larg_x,
            alt_k, larg_k,
            alt_pad, larg_pad,
            alt_s, larg_s
        );

        mm.DST = y_lote;

        cpu_matmul(&mm);
    }

    free(col);
}

static bool _usar_im2col_fw(const conv2d_fwd_params_t* params) {
    const int alt_s  = params->alt_x  + 2 * params->alt_pad  - params->alt_k  + 1;
    const int larg_s = params->larg_x + 2 * params->larg_pad - params->larg_k + 1;

    const long lotes = params->lotes;
    const long canais = params->canais;
    const long filtros = params->filtros;
    const long alt_k = params->alt_k;
    const long larg_k = params->larg_k;
    
    if (filtros < 16) return false;    
    if (alt_k < 3 && larg_k < 3) return false;
    if (alt_s * larg_s < 64) return false;

    const long flops = 2L * lotes * filtros * alt_s * larg_s * canais * alt_k * larg_k;
    const long limiar = 1e7;
    
    return flops > limiar;
}

void cpu_conv2d_forward(const conv2d_fwd_params_t* params) {
    if (_usar_im2col_fw(params)) {
        _forward_im2col(params);
    } else {
        _forward_loops(params);
    }
}

// backward

static void _backward_gk_loops(const conv2d_bwd_params_t* params) {
    const float* restrict X  = params->X;
    const float* restrict GS = params->GS;    
    float* restrict GK       = params->GK;

    const int lotes  = params->lotes;
    const int filtros = params->filtros;
    const int canais = params->canais;

    const int alt_x = params->alt_x;
    const int larg_x = params->larg_x;
    const int alt_k = params->alt_k;
    const int larg_k = params->larg_k;
    const int alt_pad  = params->alt_pad;
    const int larg_pad = params->larg_pad;

    const int alt_s  = alt_x  + 2 * alt_pad  - alt_k  + 1;
    const int larg_s = larg_x + 2 * larg_pad - larg_k + 1;

    const int area_x  = alt_x * larg_x;
    const int area_k  = alt_k * larg_k;
    const int area_gs = alt_s * larg_s;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int f = 0; f < filtros; f++) {
        for (int c = 0; c < canais; c++) {
            const int off_k_base = (f * canais + c) * area_k;

            for (int kh = 0; kh < alt_k; kh++) {
                for (int kw = 0; kw < larg_k; kw++) {
                    int i_min = MAX_ENTRE(0, alt_pad - kh);
                    int i_max = MIN_ENTRE(alt_s, alt_x + alt_pad - kh);
                    int j_min = MAX_ENTRE(0, larg_pad - kw);
                    int j_max = MIN_ENTRE(larg_s, larg_x + larg_pad - kw);
                    float soma = 0.0f;

                    for (int l = 0; l < lotes; l++) {
                        const float* restrict ptr_gs_base = GS + (l * filtros + f) * area_gs;
                        const float* restrict ptr_x_base  = X  + (l * canais + c) * area_x;

                        for (int i = i_min; i < i_max; i++) {
                            const int in_y = i + kh - alt_pad;
                            const float* restrict lin_gs = ptr_gs_base + i * larg_s;
                            const float* restrict lin_x  = ptr_x_base  + in_y * larg_x;
                            const int offset_x = kw - larg_pad; 

                            #pragma omp simd
                            for (int j = j_min; j < j_max; j++) {
                                soma += lin_gs[j] * lin_x[j + offset_x];
                            }
                        }
                    }
                    GK[off_k_base + kh * larg_k + kw] += soma;
                }
            }
        }
    }    
}

static void _backward_gk_im2col(const conv2d_bwd_params_t* params) {
    const float* restrict X  = params->X;
    const float* restrict GS = params->GS;    
    float* restrict GK       = params->GK;

    const int alt_x = params->alt_x;
    const int larg_x = params->larg_x;
    const int alt_k = params->alt_k;
    const int larg_k = params->larg_k;
    const int alt_pad  = params->alt_pad;
    const int larg_pad = params->larg_pad;
    const int alt_s  = alt_x  + 2 * alt_pad  - alt_k  + 1;
    const int larg_s = larg_x + 2 * larg_pad - larg_k + 1;

    const int lotes  = params->lotes;
    const int filtros = params->filtros;
    const int canais = params->canais;

    const int area_x  = alt_x * larg_x;

    const int Kdim = canais * alt_k * larg_k;
    const int Ndim = alt_s * larg_s;
    float* colT = malloc(sizeof(float) * Ndim * Kdim);

    matmul_params_t mm = {
        .A = NULL,
        .B = colT,
        .DST = GK,
        .lin_a = filtros,
        .col_a = Ndim,
        .col_b = Kdim,
        .std_a_0 = Ndim, .std_a_1 = 1,
        .std_b_0 = Kdim, .std_b_1 = 1,
        .std_c_0 = Kdim, .std_c_1 = 1
    };

    for (int l = 0; l < lotes; l++) {
        const float* x_lote = X + l * canais * area_x;
        const float* gs_lote = GS + l * filtros * Ndim;

        // transposta pra cair no fastpath do mm
        im2col_3dT(
            x_lote,
            colT, 
            canais, 
            alt_x, larg_x, 
            alt_k, larg_k, 
            alt_pad, larg_pad, 
            alt_s, larg_s  
        );

        mm.A = (float*)gs_lote;
        cpu_matmul(&mm); 
    }

    free(colT);
}

static bool _usar_im2col_gk(const conv2d_bwd_params_t* params) {
    if (params->canais <= 4) return true;

    const long peso = 
    (long)params->filtros * (params->canais * params->alt_k * params->larg_k) * (params->alt_x * params->larg_x);
    const int limiar = 1000000;
    
    return peso > limiar;
}

static void _backward_gk(const conv2d_bwd_params_t* params) {
    if (_usar_im2col_gk(params)) {
        _backward_gk_im2col(params);
    } else {
        _backward_gk_loops(params);
    }
}

void cpu_conv2d_backward(const conv2d_bwd_params_t* params) {
    const float* restrict K  = params->K;
    const float* restrict GS = params->GS;
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

    const int alt_pad  = params->alt_pad;
    const int larg_pad = params->larg_pad;

    const int alt_s  = alt_x  + 2 * alt_pad  - alt_k  + 1;
    const int larg_s = larg_x + 2 * larg_pad - larg_k + 1;

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

    _backward_gk(params);

    // grad entrada
    #pragma omp parallel for collapse(2) schedule(static)
    for (int l = 0; l < lotes; l++) {
        for (int c = 0; c < canais; c++) {
            float* restrict ptr_ge_base = GE + (l * canais + c) * area_x;

            for (int f = 0; f < filtros; f++) {
                const float* restrict ptr_gs_base = GS + (l * filtros + f) * area_gs;
                const float* restrict ptr_k_base  = K  + (f * canais + c) * area_k;

                for (int kh = 0; kh < alt_k; kh++) {
                    for (int kw = 0; kw < larg_k; kw++) {
                        const float val_k = ptr_k_base[kh * larg_k + kw];
                        int i_min = MAX_ENTRE(0, alt_pad - kh);
                        int i_max = MIN_ENTRE(alt_s, alt_x + alt_pad - kh);
                        int j_min = MAX_ENTRE(0, larg_pad - kw);
                        int j_max = MIN_ENTRE(larg_s, larg_x + larg_pad - kw);

                        for (int i = i_min; i < i_max; i++) {
                            const int in_y = i + kh - alt_pad;
                            float* restrict lin_ge = ptr_ge_base + in_y * larg_x;
                            const float* restrict lin_gs = ptr_gs_base + i * larg_s;
                            const int offset_ge = kw - larg_pad;

                            #pragma omp simd
                            for (int j = j_min; j < j_max; j++) {
                                lin_ge[j + offset_ge] += lin_gs[j] * val_k;
                            }
                        }
                    }
                }
            }
        }
    }

}
