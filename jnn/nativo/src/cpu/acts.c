#include "acts.h"

#include <math.h>

void relu(const float* restrict src, float* restrict dst, int tam) {
    for (int i = 0; i < tam; i++) {
        dst[i] = fmaxf(src[i], 0.f);
    }
}

void relud(const float* restrict x, const float* restrict g, float* restrict dst, int tam) {
    for (int i = 0; i < tam; i++) {
        dst[i] = g[i] * (x[i] > 0.f);
    }
}