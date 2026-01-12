#include "jnn_native_common.h"

JNIEXPORT void JNICALL
Java_jnn_nativo_JNNNative_matmul(
    JNIEnv* env, jclass cls,
    jdoubleArray A_arr, jint off_a, jint std_a_0, jint std_a_1,
    jdoubleArray B_arr, jint off_b, jint std_b_0, jint std_b_1,
    jdoubleArray C_arr, jint off_c, jint std_c_0, jint std_c_1,
    jint lin_a, jint col_a, jint col_b
) {
    (void) cls;

    double* restrict A = (*env)->GetPrimitiveArrayCritical(env, A_arr, NULL);
    double* restrict B = (*env)->GetPrimitiveArrayCritical(env, B_arr, NULL);
    double* restrict C = (*env)->GetPrimitiveArrayCritical(env, C_arr, NULL);

    const int bloco_col_a = 64;
    const int bloco_col_b = 64;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < lin_a; i++) {
        const int base_a = off_a + i * std_a_0;
        const int base_c = off_c + i * std_c_0;

        for (int kk = 0; kk < col_a; kk += bloco_col_a) {
            const int fim_k = (kk + bloco_col_a < col_a) ? kk + bloco_col_a : col_a;

            for (int jj = 0; jj < col_b; jj += bloco_col_b) {
                const int fim_j = (jj + bloco_col_b < col_b) ? jj + bloco_col_b : col_b;

                for (int k = kk; k < fim_k; k++) {
                    const double val_a = A[base_a + k * std_a_1];
                    const int base_b = off_b + k * std_b_0;

                    #pragma omp simd
                    for (int j = jj; j < fim_j; j++) {
                        C[base_c + j * std_c_1] += val_a * B[base_b + j * std_b_1];
                    }
                }
            }
        }
    }

    (*env)->ReleasePrimitiveArrayCritical(env, A_arr, A, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, B_arr, B, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, C_arr, C, 0);
}