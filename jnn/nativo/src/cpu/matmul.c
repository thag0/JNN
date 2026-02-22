#include "matmul.h"
#include "macros.h"

// tilling
#define BLOCO_LIN_A 32
#define BLOCO_COL_A 64
#define BLOCO_COL_B 64

static void _matmul_fastpath(
    const float* restrict A,
    const float* restrict B,
    float* restrict C,
    const int lin_a,
    const int col_a,
    const int col_b,
    const int std_a_0,
    const int std_b_0,
    const int std_c_0) {

    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < lin_a; ii += BLOCO_LIN_A) {
        for (int jj = 0; jj < col_b; jj += BLOCO_COL_B) {
            const int i_max = MIN_ENTRE(ii + BLOCO_LIN_A, lin_a);
            const int j_max = MIN_ENTRE(jj + BLOCO_COL_B, col_b);
    
            for (int kk = 0; kk < col_a; kk += BLOCO_COL_A) {
                const int k_max = MIN_ENTRE(kk + BLOCO_COL_A, col_a);

                for (int i = ii; i < i_max; i++) {
                    const int base_a = i * std_a_0;
                    const int base_c = i * std_c_0;
                    const int largura = j_max - jj;
                    float acc[BLOCO_COL_B];//acumulador local pra evitar escrever em C toda hora

                    #pragma omp simd
                    for (int t = 0; t < largura; t++) {
                        acc[t] = 0.0f;
                    }

                    for (int k = kk; k < k_max; k++) {
                        const float val_a = A[base_a + k];
                        const int base_b = k * std_b_0 + jj;

                        #pragma omp simd
                        for (int t = 0; t < largura; t++) {
                            acc[t] += val_a * B[base_b + t];
                        }
                    }

                    #pragma omp simd
                    for (int t = 0; t < largura; t++) {
                        C[base_c + jj + t] += acc[t];//acumulaçao unica
                    }
                }

            }
        }
    }

}

static void _matmul_generico(
    const float* restrict A, 
    const float* restrict B, 
    float* restrict C,
    int off_a, int off_b, int off_c,
    int lin_a, int col_a, int col_b,
    int std_a_0, int std_a_1,
    int std_b_0, int std_b_1,
    int std_c_0, int std_c_1) {

    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < lin_a; ii += BLOCO_LIN_A) {
        for (int jj = 0; jj < col_b; jj += BLOCO_COL_B) {
            const int i_max = MIN_ENTRE(ii + BLOCO_LIN_A, lin_a);
            const int j_max = MIN_ENTRE(jj + BLOCO_COL_B, col_b);
    
            for (int kk = 0; kk < col_a; kk += BLOCO_COL_A) {
                const int k_max = MIN_ENTRE(kk + BLOCO_COL_A, col_a);
    
                for (int i = ii; i < i_max; i++) {
                    const int base_a = off_a + i * std_a_0;
                    const int base_c = off_c + i * std_c_0;

                    int largura = j_max - jj;
                    float acc[BLOCO_COL_B];

                    #pragma omp simd
                    for (int t = 0; t < largura; t++) {
                        acc[t] = 0.0f;
                    }

                    for (int k = kk; k < k_max; k++) {
                        const float val_a = A[base_a + k * std_a_1];
                        const int base_b = off_b + k * std_b_0 + jj * std_b_1;

                        #pragma omp simd
                        for (int t = 0; t < largura; t++) {
                            acc[t] += val_a * B[base_b + t * std_b_1];
                        }
                    }

                    #pragma omp simd
                    for (int t = 0; t < largura; t++) {
                        C[base_c + (jj + t) * std_c_1] += acc[t];
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

    int fastpath = std_a_1 == 1 && std_b_1 == 1 && std_c_1 == 1;//contiguo row-major
    fastpath &= (off_a == 0 && off_b == 0 && off_dst == 0);//sem offset

    if (fastpath) {
        _matmul_fastpath(
            A, B, DST,
            lin_a, col_a, col_b,
            std_a_0, std_b_0, std_c_0
        ); 
    
    } else {
        _matmul_generico(
            A, B, DST, 
            off_a, off_b, off_dst, 
            lin_a, col_a, col_b, 
            std_a_0, std_a_1, 
            std_b_0, std_b_1, 
            std_c_0, std_c_1
        );
    }

}