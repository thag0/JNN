// algumas referencias
// https://dl.acm.org/doi/epdf/10.1145/3157733
// https://www.netlib.org/lapack/explore-html/dd/d09/group__gemm_ga8cad871c590600454d22564eff4fed6b.html

#include "gemm.h"
#include "arena.h"
#include "common.h"
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>

// microkernel
#include <immintrin.h>
#define MR 8
#define NR 8

// tilling
#define TILE_M 32//multiplo de MR
#define TILE_K 64
#define TILE_N 64//multiplo de NR

static void _para_row_major(
    const float* restrict src,
    float* restrict dst,
    int lin, int col,
    int ld_src) {

    const int tile = 16;

    #pragma omp parallel for collapse(2) schedule(static) proc_bind(close)
    for (int ii = 0; ii < lin; ii += tile) {
        for (int jj = 0; jj < col; jj += tile) {
            const int i_fim = MIN_ENTRE(ii + tile, lin);
            const int j_fim = MIN_ENTRE(jj + tile, col);

            for (int i = ii; i < i_fim; i++) {
                #pragma omp simd
                for (int j = jj; j < j_fim; j++) {
                    dst[j * lin + i] = src[i * ld_src + j];
                }
            }
        }
    }

}

static inline void _pack_b(
    const float* restrict B,
    float* restrict b_pack,
    int kk,
    int jj,
    int K_bloco,
    int N_bloco,
    int ldb,
    bool trans_b) {

    if (!trans_b) {
        for (int k = 0; k < K_bloco; k++) {
            const float* src = B + (kk + k) * ldb + jj;
            float* dst = b_pack + k * TILE_N;

            #pragma omp simd
            for (int j = 0; j < N_bloco; j++) {
                dst[j] = src[j];
            }
        }
    } else {
        for (int j = 0; j < N_bloco; j++) {
            const float* src = B + (jj + j) * ldb + kk;

            for (int k = 0; k < K_bloco; k++) {
                b_pack[(size_t)k * TILE_N + j] = src[k];
            }
        }
    }
}

static inline void _microkernel_8x8(
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
    __m256 c4 = _mm256_loadu_ps(C + 4*ldc);
    __m256 c5 = _mm256_loadu_ps(C + 5*ldc);
    __m256 c6 = _mm256_loadu_ps(C + 6*ldc);
    __m256 c7 = _mm256_loadu_ps(C + 7*ldc);

    const float* restrict ptr_a = A;
    const float* restrict ptr_b = B;

    for (int k = 0; k < K; k++) {
        __m256 b = _mm256_loadu_ps(ptr_b);

        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 0*lda), b, c0);
        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 1*lda), b, c1);
        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 2*lda), b, c2);
        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 3*lda), b, c3);
        c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 4*lda), b, c4);
        c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 5*lda), b, c5);
        c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 6*lda), b, c6);
        c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 7*lda), b, c7);

        ptr_a += 1;
        ptr_b += ldb;
    }

    _mm256_storeu_ps(C + 0*ldc, c0);
    _mm256_storeu_ps(C + 1*ldc, c1);
    _mm256_storeu_ps(C + 2*ldc, c2);
    _mm256_storeu_ps(C + 3*ldc, c3);
    _mm256_storeu_ps(C + 4*ldc, c4);
    _mm256_storeu_ps(C + 5*ldc, c5);
    _mm256_storeu_ps(C + 6*ldc, c6);
    _mm256_storeu_ps(C + 7*ldc, c7);
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
        const int base_a = i * lda;
        const int base_c = i * ldc;

        for (int j = 0; j < N; j++) {
            float acc = C[base_c + j];

            #pragma omp simd reduction(+:acc)
            for (int k = 0; k < K; k++) {
                acc += A[base_a + k] * B[k*ldb + j];
            }

            C[base_c + j] = acc;
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
    int ldc,
    bool trans_b) {

    #pragma omp parallel for collapse(2) schedule(static) proc_bind(close)
    for (int ii = 0; ii < M; ii += TILE_M) {
        for (int jj = 0; jj < N; jj += TILE_N) {
            const int M_bloco = MIN_ENTRE(TILE_M, M - ii);
            const int N_bloco = MIN_ENTRE(TILE_N, N - jj);
            float* restrict C_bloco = C + ii * ldc + jj;
            float b_pack[TILE_K * TILE_N];

            for (int kk = 0; kk < K; kk += TILE_K) {
                const int K_bloco = MIN_ENTRE(TILE_K, K - kk);

                _pack_b(
                    B, b_pack,
                    kk, jj,
                    K_bloco, N_bloco,
                    ldb,
                    trans_b
                );

                const float* restrict A_bloco =  A + ii * lda + kk;

                for (int i = 0; i < M_bloco; i += MR) {
                    const int _M = MIN_ENTRE(MR, M_bloco - i);
                    const float* restrict ptr_a = A_bloco + i * lda;

                    for (int j = 0; j < N_bloco; j += NR) {
                        const int _N = MIN_ENTRE(NR, N_bloco - j);
                        const float* restrict ptr_b = b_pack + j;
                        float* restrict ptr_c = C_bloco + i * ldc + j;

                        if (_M == MR && _N == NR) {
                            _microkernel_8x8(
                                ptr_a, ptr_b, ptr_c,
                                K_bloco,
                                lda, TILE_N, ldc
                            );
                        } else {
                            _kernel_scalar(
                                ptr_a, ptr_b, ptr_c,
                                _M, _N, K_bloco,
                                lda, TILE_N, ldc
                            );
                        }
                    }
                }
            }
        }
    }
    
}

void cpu_gemm(
    const float* restrict A,
    const float* restrict B,
    float* restrict C,
    int lda, 
    int ldb, 
    int ldc, 
    int M,
    int K,
    int N,
    bool trans_a,
    bool trans_b,
    float beta) {

    // melhorar isso aqui depois pra evitar passar duas vezes por C
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        #pragma omp simd
        for (int j = 0; j < N; j++) {
            C[(size_t)i * ldc + j] *= beta;
        }
    }

    if (trans_a) {
        const float* restrict ptr_a = A;    
        size_t checkpoint = arena_checkpoint(&mem_arena);
        float* temp_a = arena_alloc(&mem_arena, sizeof(float) * M * K);

        _para_row_major(A, temp_a, K, M, lda);
        ptr_a = temp_a;
        lda = K;

        gemm(
            ptr_a, B, C,
            M, K, N,
            lda, ldb, ldc,
            trans_b
        );

        arena_restore(&mem_arena, checkpoint);
        return;
    }

    gemm(
        A, B, C, 
        M, K, N,
        lda, ldb, ldc,
        trans_b
    );

}