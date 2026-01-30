#include "matmul.h"

// tilling
#define BLOCO_COL_A 64
#define BLOCO_COL_B 64

void _matmul_fastpath(
    const float* restrict A, 
    const float* restrict B, 
    float* restrict C,
    const int off_a, 
    const int off_b, 
    const int off_c,
    const int lin_a, 
    const int col_a, 
    const int col_b,
    const int std_a_0, 
    const int std_b_0, 
    const int std_c_0) {
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < lin_a; i++) {
        const int baseA = off_a + i * std_a_0;
        const int baseC = off_c + i * std_c_0;

        for (int kk = 0; kk < col_a; kk += BLOCO_COL_A) {
            const int kEnd = (kk + BLOCO_COL_A < col_a) ? kk + BLOCO_COL_A : col_a;

            for (int jj = 0; jj < col_b; jj += BLOCO_COL_B) {
                const int jEnd = (jj + BLOCO_COL_B < col_b) ? jj + BLOCO_COL_B : col_b;

                for (int k = kk; k < kEnd; k++) {
                    const float valA = A[baseA + k];
                    const int baseB = off_b + k * std_b_0;

                    #pragma omp simd
                    for (int j = jj; j < jEnd; j++) {
                        C[baseC + j] += valA * B[baseB + j];
                    }
                }
            }
        }
    }

}

void _matmul_generico(
    const float* restrict A, 
    const float* restrict B, 
    float* restrict C,
    int off_a, int off_b, int off_c,
    int lin_a, int col_a, int col_b,
    int std_a_0, int std_a_1,
    int std_b_0, int std_b_1,
    int std_c_0, int std_c_1) {

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < lin_a; i++) {
        const int base_a = off_a + i * std_a_0;
        const int base_c = off_c + i * std_c_0;

        for (int kk = 0; kk < col_a; kk += BLOCO_COL_A) {
            const int fim_k = (kk + BLOCO_COL_A < col_a) ? kk + BLOCO_COL_A : col_a;

            for (int jj = 0; jj < col_b; jj += BLOCO_COL_B) {
                const int fim_j = (jj + BLOCO_COL_B < col_b) ? jj + BLOCO_COL_B : col_b;

                for (int k = kk; k < fim_k; k++) {
                    const float val_a = A[base_a + k * std_a_1];
                    const int base_b = off_b + k * std_b_0;

                    #pragma omp simd
                    for (int j = jj; j < fim_j; j++) {
                        C[base_c + j * std_c_1] += val_a * B[base_b + j * std_b_1];
                    }
                }
            }
        }
    } 

}

void cpu_matmul(const matmul_params_t* params) {
    const float* restrict A = params->A;
    const float* restrict B = params->B;
    float* restrict DST = params->DST;

    const int lin_a = params->lin_a;
    const int col_a = params->col_a;
    const int col_b = params->col_b;

    const int off_a = params->off_a;
    const int off_b = params->off_b;
    const int off_dst = params->off_dst;

    const int std_a_0 = params->std_a_0;
    const int std_a_1 = params->std_a_1;
    const int std_b_0 = params->std_b_0;
    const int std_b_1 = params->std_b_1;
    const int std_c_0 = params->std_c_0;
    const int std_c_1 = params->std_c_1;

    const int fastpath = std_a_1 == 1 && std_b_1 == 1 && std_c_1 == 1;

    if (fastpath) {
        _matmul_fastpath(
            A,
            B,
            DST,
            off_a,
            off_b,
            off_dst,
            lin_a,
            col_a,
            col_b,
            std_a_0,
            std_b_0,
            std_c_0
        ); 
    } else {
        _matmul_generico(
            A, 
            B,
            DST, 
            off_a, off_b, off_dst, 
            lin_a, col_a, col_b, 
            std_a_0, std_a_1, 
            std_b_0, std_b_1, 
            std_c_0, std_c_1
        );
    }

}