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