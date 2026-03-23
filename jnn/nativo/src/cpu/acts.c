#include "acts.h"
#include <math.h>

void relu(float* restrict src, float* restrict dst, size_t tam) {
    for (size_t i = 0; i < tam; i++) {
        dst[i] = fmaxf(src[i], 0.f);
    }
}

void relu_d(float* restrict x, float* restrict g, float* restrict dst, size_t tam) {
    for (size_t i = 0; i < tam; i++) {
        dst[i] = g[i] * (x[i] > 0.f);
    }
}

void sigmoid(float* restrict src, float* restrict dst, size_t tam) {
    for (size_t i = 0; i < tam; i++) {
        dst[i] = 1.f / (1.f + expf(-src[i]));
    }
}

void sigmoid_d(float* restrict sig, float* restrict g, float* restrict dst, size_t tam) {
    for (size_t i = 0; i < tam; i++) {
        float s = sig[i];
        dst[i] = g[i] * (s * (1.f - s));
    }
}