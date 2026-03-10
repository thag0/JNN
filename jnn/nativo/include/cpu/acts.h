#pragma once

// Aplica a função de ativação relu em dst, usando src como base
void relu(const float* restrict src, float* restrict dst, int tam);

// Aplica a derivada da função de ativação relu em dst, usando src e x como base
void relud(const float* restrict x, const float* restrict g, float* restrict dst, int tam);