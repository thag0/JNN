#pragma once

#include <stddef.h>

// Aplica a função de ativação relu em dst, usando src como base
void relu(float* restrict src, float* restrict dst, size_t tam);

// Aplica a derivada da função de ativação relu em dst, usando src e x como base
void relu_d(float* restrict x, float* restrict g, float* restrict dst, size_t tam);

// Aplica a função de ativação sigmoid em dst, usando src como base
void sigmoid(float* restrict src, float* restrict dst, size_t tam);

// Aplica a derivada da função de ativação sigmoid em dst, usando src e x como base
void sigmoid_d(float* restrict sig, float* restrict g, float* restrict dst, size_t tam);