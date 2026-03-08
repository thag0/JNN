// algumas referencias
// https://dl.acm.org/doi/epdf/10.1145/3157733
// https://www.netlib.org/lapack/explore-html/dd/d09/group__gemm_ga8cad871c590600454d22564eff4fed6b.html

#include "gemm.h"
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>

// tilling
#define BLOCO_LIN_A 32
#define BLOCO_COL_A 64
#define BLOCO_COL_B 32

// microkernel
#include <immintrin.h>
#define MR 4
#define NR 8

static _Thread_local float* _pack_buff_data = NULL;
static _Thread_local size_t _pack_buff_tam = 0;

// auxiliar pra reutilizar memoria e evitar alocações
static float* _get_pack_mem(size_t tam_bytes) {
    if (tam_bytes > _pack_buff_tam) {
        _aligned_free(_pack_buff_data);
        size_t bytes = (tam_bytes + 63) & ~63;
        _pack_buff_data = _aligned_malloc(bytes, 64);
        _pack_buff_tam = bytes;
    }

    return _pack_buff_data;
}

static void _empacotar_matriz(
    const float* restrict X,
    float* restrict Y,
    int lin, int col,
    int std_lin, int std_col) {

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < lin; i++) {
        const float* lin_x = X + i * std_lin;
        float* lin_y = Y + i * col;

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

    for (int k = 0; k < K; k++) {
        __m256 b = _mm256_loadu_ps(B + k*ldb);

        __m256 a0 = _mm256_set1_ps(A[0*lda + k]);
        __m256 a1 = _mm256_set1_ps(A[1*lda + k]);
        __m256 a2 = _mm256_set1_ps(A[2*lda + k]);
        __m256 a3 = _mm256_set1_ps(A[3*lda + k]);

        c0 = _mm256_fmadd_ps(a0, b, c0);
        c1 = _mm256_fmadd_ps(a1, b, c1);
        c2 = _mm256_fmadd_ps(a2, b, c2);
        c3 = _mm256_fmadd_ps(a3, b, c3);
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

            for (int k = 0; k < K; k++) {
                acc += A[i*lda + k] * B[k*ldb + j];
            }

            C[i*ldc + j] = acc;
        }
    }
}

void _gemm(
    const float* restrict A,
    const float* restrict B,
    float* restrict C,
    int M,
    int K,
    int N,
    int lda,
    int ldb,
    int ldc) {

    #pragma omp parallel for
    for (int i = 0; i < M; i += MR) {
        int _M = (i + MR <= M) ? MR : (M - i);

        for (int j = 0; j < N; j += NR) {
            int _N = (j + NR <= N) ? NR : (N - j);
            const float* ptr_a = A + i * lda;
            const float* ptr_b = B + j;
            float* ptr_c = C + i * ldc + j;

            if (_M == MR && _N == NR) {
                _microkernel_4x8(
                    ptr_a, ptr_b, ptr_c,
                    K,
                    lda, ldb, ldc
                );
            } else {
                _kernel_scalar(
                    ptr_a, ptr_b, ptr_c,
                    _M, _N, K,
                    lda, ldb, ldc
                );
            }
        }
    }
    
}


void cpu_gemm(const gemm_params_t* params) {
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
        _gemm(
            A, B, DST, 
            lin_a, col_a, col_b,
            std_a_0, std_b_0, std_c_0
        );

    } else {
        // se precisar, empacota as matrizes pra cair no fastpath
        // como tem um buffer de memoria, as alocações nao sao um problema
        // mas ai depende do tamanho da matriz

        const float* restrict novo_A = A + off_a;
        const float* restrict novo_B = B + off_b;
        float* restrict       novo_C = DST + off_dst;

        float* restrict ptr_a = NULL;
        float* restrict ptr_b = NULL;

        int novo_std_a_0 = std_a_0;
        int novo_std_b_0 = std_b_0;
        int novo_std_c_0 = std_c_0;
    
        if (std_a_1 != 1) {
            ptr_a = _get_pack_mem(sizeof(float) * lin_a * col_a);
            _empacotar_matriz(novo_A, ptr_a, lin_a, col_a, std_a_0, std_a_1);
            novo_A = ptr_a;
            novo_std_a_0 = col_a;
        }
    
        if (std_b_1 != 1) {
            ptr_b = (float*) _aligned_malloc(sizeof(float) * col_a * col_b, 64);
            _empacotar_matriz(novo_B, ptr_b, col_a, col_b, std_b_0, std_b_1);
            novo_B = ptr_b;
            novo_std_b_0 = col_b;
        }
    
        _gemm(
            novo_A, novo_B, novo_C,
            lin_a, col_a, col_b,
            novo_std_a_0, novo_std_b_0, novo_std_c_0
        );
    
        if (ptr_b) _aligned_free(ptr_b);
    }

}