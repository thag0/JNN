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
                int row = (c * alt_k + kh) * larg_k + kw;
                float* restrict col_ptr = COL + row * area_s;
                memset(col_ptr, 0, area_s * sizeof(float));

                int h_start = MAX_ENTRE(0, alt_pad - kh);
                int h_end = MIN_ENTRE(alt_s, alt_x + alt_pad - kh);
                int w_start = MAX_ENTRE(0, larg_pad - kw);
                int w_end = MIN_ENTRE(larg_s, larg_x + larg_pad - kw);

                for (int i = h_start; i < h_end; i++) {
                    int in_y = i + kh - alt_pad;
                    const float* restrict x_ptr = X + (c * alt_x * larg_x) + (in_y * larg_x);
                    float* restrict dst_ptr = col_ptr + (i * larg_s);

                    for (int j = w_start; j < w_end; j++) {
                        int in_x = j + kw - larg_pad;
                        dst_ptr[j] = x_ptr[in_x];
                    }
                }
            }
        }
    }
}