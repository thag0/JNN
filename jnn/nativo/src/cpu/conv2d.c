#include "conv2d.h"
#include "gemm.h"
#include "im2col.h"
#include "arena.h"
#include "common.h"

#include <omp.h>

// forward

static void _forward_loops(const conv2d_fwd_params_t* params) {
    const float* restrict X = params->X;
    const float* restrict K = params->K;
    const float* restrict B = params->B;
    float* restrict DST = params->DST;

    const int alt_s  = (params->alt_x  + 2 * params->alt_pad  - params->alt_k ) + 1;
    const int larg_s = (params->larg_x + 2 * params->larg_pad - params->larg_k) + 1;

    const int area_x = params->alt_x * params->larg_x;
    const int area_k = params->alt_k * params->larg_k;
    const int area_s = alt_s * larg_s;

    #pragma omp parallel for collapse(2) schedule(static) proc_bind(close)
    for (int l = 0; l < params->lotes; l++) {
        for (int f = 0; f < params->filtros; f++) {
            float* restrict dst_base = DST + (l * params->filtros + f) * area_s;

            float bias = params->temBias ? B[f] : 0.0f;
            #pragma omp simd
            for (int i = 0; i < area_s; i++) {
                dst_base[i] = bias;
            }

            for (int c = 0; c < params->canais; c++) {
                const float* restrict Xc = X + (l * params->canais + c) * area_x;
                const float* restrict Kc = K + (f * params->canais + c) * area_k;

                for (int kh = 0; kh < params->alt_k; kh++) {
                    const int i_max = MIN_ENTRE(params->alt_x + params->alt_pad - kh, alt_s);
                    const int i_min = MAX_ENTRE(params->alt_pad - kh, 0);
                    
                    for (int kw = 0; kw < params->larg_k; kw++) {
                        const float val_k = Kc[kh * params->larg_k + kw];
                        const int j_max = MIN_ENTRE(params->larg_x + params->larg_pad - kw, larg_s);
                        const int j_min = MAX_ENTRE(params->larg_pad - kw, 0);

                        for (int i = i_min; i < i_max; i++) {
                            const int in_y = i + kh - params->alt_pad;
                            float* restrict ptr_dst = dst_base + i * larg_s;
                            const float* restrict ptr_x = Xc + in_y * params->larg_x;

                            #pragma omp simd
                            for (int j = j_min; j < j_max; j++) {
                                int in_x = j + kw - params->larg_pad;
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

    const int alt_s  = params->alt_x  + 2 * params->alt_pad  - params->alt_k  + 1;
    const int larg_s = params->larg_x + 2 * params->larg_pad - params->larg_k + 1;

    const int K_dim = params->canais * params->alt_k * params->larg_k;
    const int N_dim = alt_s * larg_s;

    size_t checkpoint = arena_checkpoint(&mem_arena);
    float* restrict col = arena_alloc(&mem_arena, sizeof(float) * K_dim * N_dim);

    const int C_x_AreaX = params->canais * params->alt_x * params->larg_x;// cache 
    const int filtros_x_Ndim = params->filtros * N_dim;// cache
    
    for (int l = 0; l < params->lotes; l++) {
        const float* restrict x_lote = X + l * C_x_AreaX;
        float* y_lote = DST + l * filtros_x_Ndim;

        for (int f = 0; f < params->filtros; f++) {
            float bias = params->temBias ? B[f] : 0.f;
            float* restrict linha = y_lote + f * N_dim;
            for (int i = 0; i < N_dim; i++) {
                linha[i] = bias;
            }
        }

        im2col(
            x_lote, col,
            params->canais,
            params->alt_x, params->larg_x,
            params->alt_k, params->larg_k,
            params->alt_pad, params->larg_pad,
            alt_s, larg_s
        );

        cpu_gemm(
            false, 
            false,
            params->filtros, 
            N_dim,
            K_dim, 
            K,
            K_dim,
            col,
            N_dim,
            1.0f,
            y_lote,
            N_dim
        );
    }

    arena_restore(&mem_arena, checkpoint);
}

static bool _usar_im2col_fw(const conv2d_fwd_params_t* params) {
    const long alt_s = (long)params->alt_x + 2L * params->alt_pad - params->alt_k + 1;
    const long larg_s = (long)params->larg_x + 2L * params->larg_pad - params->larg_k + 1;
    const long M = params->filtros;
    const long K = (long)params->canais * params->alt_k * params->larg_k;
    const long N = alt_s * larg_s;
    
    const long operacoes = M * K * N;
    const long custo_im2col = K * N;

    long limiar;
    if (M >= 64) {
        limiar = 250000L;
    } else if (M >= 32) {
        limiar = 500000L;
    } else if (M >= 16) {
        limiar = 900000L;
    } else if (M >= 8) {
        limiar = 1600000L;
    } else {
        limiar = 2800000L;
    }

    if (params->alt_k * params->larg_k >= 9) limiar = limiar * 3L / 4L;
    if (N < 16) return false;
    if (M < 4 && custo_im2col > 4096L) return false;
    if (N >= 256) limiar = limiar * 3L / 4L;
    if (K < 16 && M < 32) limiar = limiar * 2L;

    return operacoes > limiar;
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

    const int alt_s  = params->alt_x  + 2 * params->alt_pad  - params->alt_k  + 1;
    const int larg_s = params->larg_x + 2 * params->larg_pad - params->larg_k + 1;

    const int area_x  = params->alt_x * params->larg_x;
    const int area_k  = params->alt_k * params->larg_k;
    const int area_gs = alt_s * larg_s;

    #pragma omp parallel for collapse(2) schedule(static) proc_bind(close)
    for (int f = 0; f < params->filtros; f++) {
        for (int c = 0; c < params->canais; c++) {
            const int off_k_base = (f * params->canais + c) * area_k;

            for (int kh = 0; kh < params->alt_k; kh++) {
                int i_min = MAX_ENTRE(0, params->alt_pad - kh);
                int i_max = MIN_ENTRE(alt_s, params->alt_x + params->alt_pad - kh);
                
                for (int kw = 0; kw < params->larg_k; kw++) {
                    int j_min = MAX_ENTRE(0, params->larg_pad - kw);
                    int j_max = MIN_ENTRE(larg_s, params->larg_x + params->larg_pad - kw);
                    float soma = 0.0f;

                    for (int l = 0; l < params->lotes; l++) {
                        const float* restrict ptr_gs_base = GS + (l * params->filtros + f) * area_gs;
                        const float* restrict ptr_x_base  = X  + (l * params->canais + c) * area_x;

                        for (int i = i_min; i < i_max; i++) {
                            const int in_y = i + kh - params->alt_pad;
                            const float* restrict lin_gs = ptr_gs_base + i * larg_s;
                            const float* restrict lin_x  = ptr_x_base  + in_y * params->larg_x;
                            const int offset_x = kw - params->larg_pad; 

                            #pragma omp simd reduction(+:soma)
                            for (int j = j_min; j < j_max; j++) {
                                soma += lin_gs[j] * lin_x[j + offset_x];
                            }
                        }
                    }
                    GK[off_k_base + kh * params->larg_k + kw] += soma;
                }
            }
        }
    }    
}

static void _backward_gk_im2col(const conv2d_bwd_params_t* params) {
    const float* restrict X  = params->X;
    const float* restrict GS = params->GS;    
    float* restrict GK       = params->GK;

    const int alt_s  = params->alt_x  + 2 * params->alt_pad  - params->alt_k  + 1;
    const int larg_s = params->larg_x + 2 * params->larg_pad - params->larg_k + 1;

    const int K_dim = params->canais * params->alt_k * params->larg_k;
    const int N_dim = alt_s * larg_s;
    
    size_t checkpoint = arena_checkpoint(&mem_arena);
    float* restrict col = arena_alloc(&mem_arena, sizeof(float) * K_dim * N_dim);

    for (int l = 0; l < params->lotes; l++) {
        const float* restrict x_lote  = X  + l * (params->canais * params->alt_x * params->larg_x);
        const float* restrict gs_lote = GS + l * (params->filtros * N_dim);

        im2col(
            x_lote, col,
            params->canais,
            params->alt_x, params->larg_x,
            params->alt_k, params->larg_k,
            params->alt_pad, params->larg_pad,
            alt_s, larg_s
        );

        cpu_gemm(
            false, true,
            params->filtros, K_dim, N_dim,
            gs_lote, N_dim, 
            col, K_dim,
            1.0f,
            GK, K_dim
        );
    }

    arena_restore(&mem_arena, checkpoint);
}

static bool _usar_im2col_gk(const conv2d_bwd_params_t* params) {
    if (params->canais <= 4) return false;

    const long peso = 
    (long)params->filtros * 
    params->canais * 
    params->alt_k * 
    params->larg_k * 
    params->alt_x * 
    params->larg_x;

    const int limiar = 1500000;
    
    return peso > limiar;
}

static void _backward_gk(const conv2d_bwd_params_t* params) {
    if (_usar_im2col_gk(params)) {
        _backward_gk_im2col(params);
    } else {
        _backward_gk_loops(params);
    }
}

static void _backward_ge_loops(const conv2d_bwd_params_t* params) {
    const int alt_s  = params->alt_x  + 2 * params->alt_pad  - params->alt_k  + 1;
    const int larg_s = params->larg_x + 2 * params->larg_pad - params->larg_k + 1;

    const int area_x  = params->alt_x * params->larg_x;
    const int area_k  = params->alt_k * params->larg_k;
    const int area_gs = alt_s * larg_s;

    #pragma omp parallel for collapse(2) schedule(static) proc_bind(close)
    for (int l = 0; l < params->lotes; l++) {
        for (int c = 0; c < params->canais; c++) {
            float* restrict ptr_ge_base = params->GE + (l * params->canais + c) * area_x;

            for (int f = 0; f < params->filtros; f++) {
                const float* restrict ptr_gs_base = params->GS + (l * params->filtros + f) * area_gs;
                const float* restrict ptr_k_base  = params->K  + (f * params->canais + c) * area_k;

                for (int kh = 0; kh < params->alt_k; kh++) {
                    int i_min = MAX_ENTRE(0, params->alt_pad - kh);
                    int i_max = MIN_ENTRE(alt_s, params->alt_x + params->alt_pad - kh);
                    
                    for (int kw = 0; kw < params->larg_k; kw++) {
                        const int offset_ge = kw - params->larg_pad;
                        const float val_k = ptr_k_base[kh * params->larg_k + kw];
                        int j_min = MAX_ENTRE(0, params->larg_pad - kw);
                        int j_max = MIN_ENTRE(larg_s, params->larg_x + params->larg_pad - kw);

                        for (int i = i_min; i < i_max; i++) {
                            const int in_y = i + kh - params->alt_pad;
                            float* restrict lin_ge = ptr_ge_base + in_y * params->larg_x;
                            const float* restrict lin_gs = ptr_gs_base + i * larg_s;

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

static void _flip_conv_kernel(const float* restrict K, float* restrict k_flip, int filtros, int canais, int alt_k, int larg_k) {
    const int area_k = alt_k * larg_k;

    for (int c = 0; c < canais; c++) {
        float* restrict dst_c = k_flip + (size_t)c * filtros * area_k;
        
        for (int f = 0; f < filtros; f++) {
            const float* restrict src = K + ((size_t)f * canais + c) * area_k;
            float* restrict dst = dst_c + (size_t)f * area_k;
            
            for (int t = 0; t < area_k; t++) {
                dst[t] = src[area_k - 1 - t];
            }
        }
    }
}

static void _backward_ge_conv(const conv2d_bwd_params_t* params) {
    const int alt_s  = params->alt_x  + 2*params->alt_pad  - params->alt_k  + 1;
    const int larg_s = params->larg_x + 2*params->larg_pad - params->larg_k + 1;
    
    const int pad_h  = params->alt_k  - 1 - params->alt_pad;
    const int pad_w  = params->larg_k - 1 - params->larg_pad;
    
    const int K_dim = params->filtros * params->alt_k * params->larg_k;
    const int N_dim = params->alt_x * params->larg_x;
    const int threads = omp_get_max_threads();
    
    size_t checkpoint = arena_checkpoint(&mem_arena);
    float* col_full       = arena_alloc(&mem_arena, sizeof(float) * (size_t)threads * K_dim * N_dim);
    float* kernel_flipped = arena_alloc(&mem_arena, sizeof(float) * (size_t)params->canais * K_dim);
    _flip_conv_kernel(params->K, kernel_flipped, params->filtros, params->canais, params->alt_k, params->larg_k);

    #pragma omp parallel for schedule(static) proc_bind(close)
    for (int l = 0; l < params->lotes; l++) {
        float* col = col_full + (size_t)omp_get_thread_num() * K_dim * N_dim;
        const float* gs_lote = params->GS + (size_t)l * params->filtros * alt_s * larg_s;
        float* ge_lote       = params->GE + (size_t)l * params->canais * N_dim;

        im2col_st(
            gs_lote, col,
            params->filtros,
            alt_s, larg_s,
            params->alt_k, params->larg_k,
            pad_h, pad_w,
            params->alt_x, params->larg_x
        );

        cpu_gemm_st(
            false, false, 
            params->canais, N_dim, K_dim,
            kernel_flipped, K_dim,
            col, N_dim,
            1.0f,
            ge_lote, N_dim
        );
    }

    arena_restore(&mem_arena, checkpoint);
}

static bool _usar_col2im_ge(const conv2d_bwd_params_t* params) {
    if (params->canais <= 4) return false;
    if (params->filtros <= 16) return false;
    
    long im2col_size = 
    (long) params->lotes *
    params->canais *
    params->filtros *
    params->alt_k *
    params->larg_k *
    params->larg_x *
    params->alt_x;

    const long limiar = 3200000;

    return im2col_size > limiar;
}

static void _backward_ge(const conv2d_bwd_params_t* params) {
    if (_usar_col2im_ge(params)) {
        _backward_ge_conv(params);
    } else {
        _backward_ge_loops(params);
    }
}

void cpu_conv2d_backward(const conv2d_bwd_params_t* params) {
    const float* restrict GS = params->GS;
    float* restrict GB       = params->GB;

    const int alt_s  = params->alt_x  + 2 * params->alt_pad  - params->alt_k  + 1;
    const int larg_s = params->larg_x + 2 * params->larg_pad - params->larg_k + 1;
    const int area_gs = alt_s * larg_s;

    if (params->temBias) {
        #pragma omp parallel for schedule(static) proc_bind(close)
        for (int f = 0; f < params->filtros; f++) {
            float soma_bias = 0.0f;
            const int f_offset = f * area_gs;
            for (int l = 0; l < params->lotes; l++) {
                const float* restrict ptr_gs = GS + l * params->filtros * area_gs + f_offset;

                #pragma omp simd reduction(+:soma_bias)
                for (int i = 0; i < area_gs; i++) {
                    soma_bias += ptr_gs[i];
                }
            }

            GB[f] += soma_bias;
        }
    }

    _backward_gk(params);

    _backward_ge(params);
}
