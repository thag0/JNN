#include "im2col.h"

#include <string.h>
#include <stdbool.h>
#include "common.h"

void im2col(
    const float* restrict X,
    float* restrict COL,
    int canais,
    int alt_x, int larg_x,
    int alt_k, int larg_k,
    int alt_pad, int larg_pad,
    int alt_s, int larg_s) {
        
    const int Ndim = alt_s * larg_s;
    
    #pragma omp parallel for collapse(3) schedule(static) proc_bind(close)
    for (int c = 0; c < canais; c++) {
        for (int kh = 0; kh < alt_k; kh++) {
            for (int kw = 0; kw < larg_k; kw++) {
                int linha = (c * alt_k + kh) * larg_k + kw;
                float* restrict ptr_col = COL + (size_t)linha * Ndim;

                int h_min = MAX_ENTRE(0, alt_pad - kh);
                int h_max = MIN_ENTRE(alt_s, alt_x + alt_pad - kh);
                int w_min = MAX_ENTRE(0, larg_pad - kw);
                int w_max = MIN_ENTRE(larg_s, larg_x + larg_pad - kw);
                int largura = w_max - w_min;
                int in_x = w_min + kw - larg_pad;

                if (h_min > 0) memset(ptr_col, 0, sizeof(float) * h_min * larg_s);
                if (h_max < alt_s) memset(ptr_col + h_max * larg_s, 0, sizeof(float) * (alt_s - h_max) * larg_s);

                for (int i = h_min; i < h_max; i++) {
                    int in_y = i + kh - alt_pad;
                    const float* restrict ptr_x = X + (c * alt_x * larg_x) + in_y * larg_x;
                    const float* restrict x = ptr_x + in_x;
                    float* restrict dest_row = ptr_col + i * larg_s;

                    if (w_min > 0) memset(dest_row, 0, sizeof(float) * w_min);
                    if (w_max < larg_s) memset(dest_row + w_max, 0, sizeof(float) * (larg_s - w_max));

                    float* restrict dest = dest_row + w_min;
                    #pragma omp simd
                    for (int t = 0; t < largura; t++) {
                        dest[t] = x[t];
                    }
                }
            }
        }
    }

}

void col2im_T(
    const float* restrict COLT,
    float* restrict GE,
    int canais,
    int alt_x, int larg_x,
    int alt_k, int larg_k,
    int alt_pad, int larg_pad,
    int alt_s, int larg_s) {

    const int Kdim = canais * alt_k * larg_k;
    const int area_x = alt_x * larg_x;

    #pragma omp parallel for schedule(static) proc_bind(close)
    for (int c = 0; c < canais; c++) {
        const int base_k_c = c * alt_k * larg_k;
        float* restrict ge_c = GE + c * area_x;

        for (int i = 0; i < alt_s; i++) {
            const int kh_min = MAX_ENTRE(0, alt_pad - i);
            const int kh_max = MIN_ENTRE(alt_k, alt_x + alt_pad - i);

            for (int j = 0; j < larg_s; j++) {
                const int kw_min = MAX_ENTRE(0, larg_pad - j);
                const int kw_max = MIN_ENTRE(larg_k, larg_x + larg_pad - j);

                const int n = i * larg_s + j;
                const float* restrict lin_col = COLT + n * Kdim + base_k_c;

                for (int kh = kh_min; kh < kh_max; kh++) {
                    const int in_y = i + kh - alt_pad;
                    float* restrict lin_ge = ge_c + in_y * larg_x;
                    const float* restrict lin_col_kh = lin_col + kh * larg_k;

                    #pragma omp simd
                    for (int kw = kw_min; kw < kw_max; kw++) {
                        const int in_x = j + kw - larg_pad;
                        lin_ge[in_x] += lin_col_kh[kw];
                    }
                }
            }
        }
    }

}