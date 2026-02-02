#pragma once

typedef struct {
    float* restrict A;
    float* restrict B;
    float* restrict DST;

    int off_a, off_b, off_dst;

    int std_a_0, std_a_1;
    int std_b_0, std_b_1;
    int std_c_0, std_c_1;

    int lin_a;
    int col_a;
    int col_b;
} matmul_params_t;

// Realiza a multiplicação matricial entre A e B, salvando de DST.
void cpu_matmul(const matmul_params_t* params);