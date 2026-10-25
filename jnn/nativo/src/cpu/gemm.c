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

//
#define PARA_CADA_ACC(M) M(0); M(1); M(2); M(3); M(4); M(5); M(6); M(7)
#define ZERO_ACC(i) c##i = _mm256_setzero_ps()
#define LOAD_ACC(i) c##i = _mm256_loadu_ps(C + (i) * ldc)
#define MUL_ACC(i) c##i = _mm256_mul_ps(_mm256_loadu_ps(C + (i) * ldc), vb)
#define UPDATE_ACC(i) c##i = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + (i)), b, c##i)
#define STORE_ACC(i) _mm256_storeu_ps(C + (i) * ldc, c##i)

static inline void _microkernel_8x8(
    const float* restrict A,
    const float* restrict B,
    float* restrict C,
    int K,
    int ldb,
    int ldc,
    float beta) {

    __m256 c0, c1, c2, c3, c4, c5, c6, c7;

    if (beta == 0.0f) {
        PARA_CADA_ACC(ZERO_ACC);
    
    } else if (beta == 1.0f) {
        PARA_CADA_ACC(LOAD_ACC);
    
    } else {
        const __m256 vb = _mm256_set1_ps(beta);
        PARA_CADA_ACC(MUL_ACC);
    }

    const float* restrict ptr_a = A;
    const float* restrict ptr_b = B;

    for (int k = 0; k < K; k++) {
        const __m256 b = _mm256_load_ps(ptr_b);

        PARA_CADA_ACC(UPDATE_ACC);

        ptr_a += MR;
        ptr_b += ldb;
    }

    PARA_CADA_ACC(STORE_ACC);
}
//

static inline void _pack_a(
    const float* restrict A,
    float* restrict a_pack,
    int ii,
    int kk,
    int M_bloco,
    int K_bloco,
    int lda,
    bool trans_a) {

    for (int i = 0; i < M_bloco; i += MR) {
        const int mr = MIN_ENTRE(MR, M_bloco - i);
        float* restrict dst = a_pack + (size_t)i * K_bloco;

        if (!trans_a) {
            for (int r = 0; r < mr; r++) {
                const float* src = A + (size_t)(ii + i + r) * lda + kk;

                for (int k = 0; k < K_bloco; k++) {
                    dst[k * MR + r] = src[k];
                }
            }
            for (int r = mr; r < MR; r++) {
                for (int k = 0; k < K_bloco; k++) {
                    dst[k * MR + r] = 0.0f;
                }
            }
        } else {
            for (int k = 0; k < K_bloco; k++) {
                const float* src = A + (size_t)(kk + k) * lda + ii + i;
                float* d = dst + k * MR;

                for (int r = 0; r < mr; r++) {
                    d[r] = src[r];
                }
                for (int r = mr; r < MR; r++) {
                    d[r] = 0.0f;
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

static inline void _kernel_scalar(
    const float* restrict A,
    const float* restrict B,
    float* restrict C,
    int M,
    int N,
    int K,
    int ldb,
    int ldc,
    float beta) {

    for (int i = 0; i < M; i++) {
        float* restrict c = C + i * ldc;

        if (beta == 0.0f) {
            for (int j = 0; j < N; j++) {
                c[j] = 0.0f;
            }
        } else if (beta != 1.0f) {
            for (int j = 0; j < N; j++) {
                c[j] *= beta;
            }
        }

        for (int k = 0; k < K; k++) {
            const float a = A[k * MR + i];
            const float* restrict b = B + k * ldb;

            #pragma omp simd
            for (int j = 0; j < N; j++) {
                c[j] += a * b[j];
            }
        }
    }

}

void cpu_gemm(
    bool trans_a,
    bool trans_b,
    int M,
    int N,
    int K,
    const float* restrict A,
    int lda,
    const float* restrict B,
    int ldb,
    float beta,
    float* restrict C,
    int ldc) {

    #pragma omp parallel for collapse(2) schedule(static) proc_bind(close)
    for (int ii = 0; ii < M; ii += TILE_M) {
        for (int jj = 0; jj < N; jj += TILE_N) {
            const int M_bloco = MIN_ENTRE(TILE_M, M - ii);
            const int N_bloco = MIN_ENTRE(TILE_N, N - jj);
            float* restrict C_bloco = C + (size_t)ii * ldc + jj;

            _Alignas(32) float a_pack[TILE_M * TILE_K];
            _Alignas(32) float b_pack[TILE_K * TILE_N];

            for (int kk = 0; kk < K; kk += TILE_K) {
                const int K_bloco = MIN_ENTRE(TILE_K, K - kk);
                const float beta_tile = (kk == 0) ? beta : 1.0f;

                _pack_a(A, a_pack, ii, kk, M_bloco, K_bloco, lda, trans_a);
                _pack_b(B, b_pack, kk, jj, K_bloco, N_bloco, ldb, trans_b);

                for (int i = 0; i < M_bloco; i += MR) {
                    const int _M = MIN_ENTRE(MR, M_bloco - i);
                    const float* restrict ptr_a = a_pack + (size_t)i * K_bloco;

                    for (int j = 0; j < N_bloco; j += NR) {
                        const int _N = MIN_ENTRE(NR, N_bloco - j);
                        const float* restrict ptr_b = b_pack + j;
                        float* restrict ptr_c = C_bloco + (size_t)i * ldc + j;

                        if (_M == MR && _N == NR) {
                            _microkernel_8x8(
                                ptr_a, ptr_b, ptr_c,
                                K_bloco,
                                TILE_N, ldc,
                                beta_tile
                            );
                        } else {
                            _kernel_scalar(
                                ptr_a, ptr_b, ptr_c,
                                _M, _N, K_bloco,
                                TILE_N, ldc,
                                beta_tile
                            );
                        }
                    }
                }
            }
        }
    }
    
}

void cpu_gemm_st(
    bool trans_a,
    bool trans_b,
    int M,
    int N,
    int K,
    const float* restrict A,
    int lda,
    const float* restrict B,
    int ldb,
    float beta,
    float* restrict C,
    int ldc) {

    _Alignas(32) float a_pack[TILE_M * TILE_K];
    _Alignas(32) float b_pack[TILE_K * TILE_N];

    for (int ii = 0; ii < M; ii += TILE_M) {
        const int M_bloco = MIN_ENTRE(TILE_M, M - ii);
        
        for (int jj = 0; jj < N; jj += TILE_N) {
            const int N_bloco = MIN_ENTRE(TILE_N, N - jj);
            float* restrict C_bloco = C + (size_t)ii * ldc + jj;

            for (int kk = 0; kk < K; kk += TILE_K) {
                const int K_bloco = MIN_ENTRE(TILE_K, K - kk);
                const float beta_tile = (kk == 0) ? beta : 1.0f;

                _pack_a(A, a_pack, ii, kk, M_bloco, K_bloco, lda, trans_a);
                _pack_b(B, b_pack, kk, jj, K_bloco, N_bloco, ldb, trans_b);

                for (int i = 0; i < M_bloco; i += MR) {
                    const int _M = MIN_ENTRE(MR, M_bloco - i);
                    const float* restrict ptr_a = a_pack + (size_t)i * K_bloco;

                    for (int j = 0; j < N_bloco; j += NR) {
                        const int _N = MIN_ENTRE(NR, N_bloco - j);
                        const float* restrict ptr_b = b_pack + j;
                        float* restrict ptr_c = C_bloco + (size_t)i * ldc + j;

                        if (_M == MR && _N == NR) {
                            _microkernel_8x8(
                                ptr_a, ptr_b, ptr_c,
                                K_bloco,
                                TILE_N, ldc,
                                beta_tile
                            );
                        } else {
                            _kernel_scalar(
                                ptr_a, ptr_b, ptr_c,
                                _M, _N, K_bloco,
                                TILE_N, ldc,
                                beta_tile
                            );
                        }
                    }
                }
            }
        }
    }
    
}