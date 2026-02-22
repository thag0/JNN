#include "im2col.h"
#include "macros.h"
#include <string.h>

void im2col_3d(
    const float* restrict X,
    float* restrict COL,
    int canais,
    int alt_x, int larg_x,
    int alt_k, int larg_k,
    int alt_pad, int larg_pad,
    int alt_s, int larg_s) {

    const int area_s = alt_s * larg_s;
    const int Kdim = canais * alt_k * larg_k;

    //tem que zerar porque pode vir lixo do malloc, principalemnte com padding
    memset(COL, 0, sizeof(float) * Kdim * area_s);

    #pragma omp parallel for collapse(3) schedule(static)
    for (int c = 0; c < canais; c++) {
        for (int kh = 0; kh < alt_k; kh++) {
            for (int kw = 0; kw < larg_k; kw++) {
                int linha = (c * alt_k + kh) * larg_k + kw;
                float* restrict ptr_col = COL + linha * area_s;
                int h_min = MAX_ENTRE(0, alt_pad - kh);
                int h_max = MIN_ENTRE(alt_s, alt_x + alt_pad - kh);
                int w_min = MAX_ENTRE(0, larg_pad - kw);
                int w_max = MIN_ENTRE(larg_s, larg_x + larg_pad - kw);

                for (int i = h_min; i < h_max; i++) {
                    int in_y = i + kh - alt_pad;
                    const float* restrict ptr_x = X + (c * alt_x * larg_x) + (in_y * larg_x);
                    float* restrict ptr_dst = ptr_col + (i * larg_s);

                    int in_x = w_min + kw - larg_pad;
                    int largura = w_max - w_min;
                    const float* x = ptr_x + in_x;
                    float* dest = ptr_dst + w_min;

                    #pragma omp simd
                    for (int t = 0; t < largura; t++) {
                        dest[t] = x[t];
                    }
                }
            }
        }
    }
}

void im2col_3dT(
    const float* restrict X,
    float* restrict COLT,
    int canais,
    int alt_x, int larg_x,
    int alt_k, int larg_k,
    int alt_pad, int larg_pad,
    int alt_s, int larg_s) {

    const int Kdim = canais * alt_k * larg_k;
    const int Ndim = alt_s * larg_s;

    memset(COLT, 0, sizeof(float) * Kdim * Ndim);//mesma coisa

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < alt_s; i++) {
        for (int j = 0; j < larg_s; j++) {
            const int n = i * larg_s + j;
            float* restrict lin_col = COLT + n * Kdim;

            for (int c = 0; c < canais; c++) {
                const int base_x_c = c * alt_x * larg_x;
                const int base_k_c = c * alt_k * larg_k;

                for (int kh = 0; kh < alt_k; kh++) {
                    int in_y = i + kh - alt_pad;
                    if ((unsigned)in_y >= (unsigned)alt_x) continue;

                    const float* restrict lin_x = X + base_x_c + in_y * larg_x;

                    for (int kw = 0; kw < larg_k; kw++) {
                        int in_x = j + kw - larg_pad;
                        if ((unsigned)in_x >= (unsigned)larg_x) continue;
                        const int k = base_k_c + kh * larg_k + kw;
                        lin_col[k] = lin_x[in_x];
                    }
                }
            }
        }
    }
}