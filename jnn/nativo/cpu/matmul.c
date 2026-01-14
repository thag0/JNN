#include "matmul.h"

void cpu_matmul(const matmul_params_t* params) {
    double* restrict A = params->A;
    double* restrict B = params->B;
    double* restrict DST = params->DST;

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

    // tilling
    const int bloco_col_a = 64;
    const int bloco_col_b = 64;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < lin_a; i++) {
        const int base_a = off_a + i * std_a_0;
        const int base_c = off_dst + i * std_c_0;

        for (int kk = 0; kk < col_a; kk += bloco_col_a) {
            const int fim_k = (kk + bloco_col_a < col_a) ? kk + bloco_col_a : col_a;

            for (int jj = 0; jj < col_b; jj += bloco_col_b) {
                const int fim_j = (jj + bloco_col_b < col_b) ? jj + bloco_col_b : col_b;

                for (int k = kk; k < fim_k; k++) {
                    const double val_a = A[base_a + k * std_a_1];
                    const int base_b = off_b + k * std_b_0;

                    #pragma omp simd
                    for (int j = jj; j < fim_j; j++) {
                        DST[base_c + j * std_c_1] += val_a * B[base_b + j * std_b_1];
                    }
                }
            }
        }
    } 
}