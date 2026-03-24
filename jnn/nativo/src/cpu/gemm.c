// algumas referencias
// https://dl.acm.org/doi/epdf/10.1145/3157733
// https://www.netlib.org/lapack/explore-html/dd/d09/group__gemm_ga8cad871c590600454d22564eff4fed6b.html

#include "gemm.h"
#include "mem_pool.h"
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>

// tilling
#define BLOCO_LIN_A 16
#define BLOCO_COL_A 64
#define BLOCO_COL_B 32

// microkernel
#include <immintrin.h>
#define MR 4
#define NR 8

static void _para_row_major(
    const float* restrict X,
    float* restrict Y,
    int lin, int col,
    int std_lin, int std_col) {

    #pragma omp parallel for schedule(static) proc_bind(close)
    for (int i = 0; i < lin; i++) {
        const float* restrict lin_x = X + i * std_lin;
        float* restrict lin_y = Y + i * col;

        #pragma omp simd
        for (int j = 0; j < col; j++) {
            lin_y[j] = lin_x[j * std_col];
        }
    }
}

static inline void _microkernel_4x8(
    const float* restrict A,
    const float* restrict B,
    float* restrict C,
    int K,
    int lda,
    int ldb,
    int ldc) {

    __m256 c0 = _mm256_loadu_ps(C + 0*ldc);
    __m256 c1 = _mm256_loadu_ps(C + 1*ldc);
    __m256 c2 = _mm256_loadu_ps(C + 2*ldc);
    __m256 c3 = _mm256_loadu_ps(C + 3*ldc);

    const float* restrict ptr_a = A;
    const float* restrict ptr_b = B;

    for (int k = 0; k < K; k++) {
        __m256 b = _mm256_loadu_ps(ptr_b);

        __m256 a0 = _mm256_broadcast_ss(ptr_a + 0*lda);
        __m256 a1 = _mm256_broadcast_ss(ptr_a + 1*lda);
        __m256 a2 = _mm256_broadcast_ss(ptr_a + 2*lda);
        __m256 a3 = _mm256_broadcast_ss(ptr_a + 3*lda);

        c0 = _mm256_fmadd_ps(a0, b, c0);
        c1 = _mm256_fmadd_ps(a1, b, c1);
        c2 = _mm256_fmadd_ps(a2, b, c2);
        c3 = _mm256_fmadd_ps(a3, b, c3);

        ptr_a += 1;
        ptr_b += ldb;
    }

    _mm256_storeu_ps(C + 0*ldc, c0);
    _mm256_storeu_ps(C + 1*ldc, c1);
    _mm256_storeu_ps(C + 2*ldc, c2);
    _mm256_storeu_ps(C + 3*ldc, c3);
}

static inline void _kernel_scalar(
    const float* restrict A,
    const float* restrict B,
    float* restrict C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc) {

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = C[i*ldc + j];
            const int base_a = i * lda;

            #pragma omp simd
            for (int k = 0; k < K; k++) {
                acc += A[base_a + k] * B[k*ldb + j];
            }

            C[i*ldc + j] = acc;
        }
    }
}

static void gemm(
    const float* restrict A,
    const float* restrict B,
    float* restrict C,
    int M,
    int K,
    int N,
    int lda,
    int ldb,
    int ldc) {

    #pragma omp parallel for collapse(2) schedule(static) proc_bind(close)
    for (int ii = 0; ii < M; ii += BLOCO_LIN_A) {
        for (int jj = 0; jj < N; jj += BLOCO_COL_B) {
            int M_bloco = (ii + BLOCO_LIN_A <= M) ? BLOCO_LIN_A : (M - ii);
            int N_bloco = (jj + BLOCO_COL_B <= N) ? BLOCO_COL_B : (N - jj);
            float* restrict C_bloco = C + ii * ldc + jj;

            for (int kk = 0; kk < K; kk += BLOCO_COL_A) {
                int K_bloco = (kk + BLOCO_COL_A <= K) ? BLOCO_COL_A : (K - kk);
                const float* restrict A_bloco = A + ii * lda + kk;
                const float* restrict B_bloco = B + kk * ldb + jj;

                for (int i = 0; i < M_bloco; i += MR) {
                    int _M = (i + MR <= M_bloco) ? MR : (M_bloco - i);
                    const float* restrict ptr_a = A_bloco + i * lda;

                    for (int j = 0; j < N_bloco; j += NR) {
                        int _N = (j + NR <= N_bloco) ? NR : (N_bloco - j);
                        const float* restrict ptr_b = B_bloco + j;
                        float* restrict ptr_c = C_bloco + i * ldc + j;

                        if (_M == MR && _N == NR) {
                            _microkernel_4x8(
                                ptr_a, ptr_b, ptr_c,
                                K_bloco,
                                lda, ldb, ldc
                            );
                        } else {
                            _kernel_scalar(
                                ptr_a, ptr_b, ptr_c,
                                _M, _N, K_bloco,
                                lda, ldb, ldc
                            );
                        }
                    }
                }
            }
        }
    }
    
}

void cpu_gemm(gemm_params_t* params) {
    const float* restrict A = params->A + params->off_a;
    const float* restrict B = params->B + params->off_b;
    float* restrict       C = params->C + params->off_c;

    const int lin_a = params->lin_a;
    const int col_a = params->col_a;
    const int col_b = params->col_b;

    const int std_a_0 = params->std_a_0;
    const int std_a_1 = params->std_a_1;
    const int std_b_0 = params->std_b_0;
    const int std_b_1 = params->std_b_1;
    const int std_c_0 = params->std_c_0;
    const int std_c_1 = params->std_c_1;

    if (std_a_1 == 1 && std_b_1 == 1 && std_c_1 == 1) {//contiguo row-major
        gemm(
            A, B, C, 
            lin_a, col_a, col_b,
            std_a_0, std_b_0, std_c_0
        );
    } else {
        int lda = std_a_0;
        int ldb = std_b_0;
        int ldc = std_c_0;

        float* restrict ptr_a = NULL;
        float* restrict ptr_b = NULL;
    
        if (std_a_1 != 1) {
            ptr_a = get_gemm_mem_pool_a(sizeof(float) * lin_a * col_a);
            _para_row_major(A, ptr_a, lin_a, col_a, std_a_0, std_a_1);
            A = ptr_a;
            lda = col_a;
        }
    
        if (std_b_1 != 1) {
            ptr_b = get_gemm_mem_pool_b(sizeof(float) * col_a * col_b);
            _para_row_major(B, ptr_b, col_a, col_b, std_b_0, std_b_1);
            B = ptr_b;
            ldb = col_b;
        }
    
        gemm(
            A, B, C,
            lin_a, col_a, col_b,
            lda, ldb, ldc
        );
    }

}