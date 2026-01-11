#include "jnn_native_common.h"

JNIEXPORT void JNICALL
Java_jnn_nativo_JNNNative_matmul(
    JNIEnv* env, jclass cls,
    jdoubleArray aArr, jint offA, jint s0A, jint s1A,
    jdoubleArray bArr, jint offB, jint s0B, jint s1B,
    jdoubleArray cArr, jint offC, jint s0C, jint s1C,
    jint linA, jint colA, jint colB
) {
    (void) cls;

    double* restrict A = (*env)->GetPrimitiveArrayCritical(env, aArr, NULL);
    double* restrict B = (*env)->GetPrimitiveArrayCritical(env, bArr, NULL);
    double* restrict C = (*env)->GetPrimitiveArrayCritical(env, cArr, NULL);

    const int blocoColA = 64;
    const int blocoColB = 64;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < linA; i++) {
        const int baseA = offA + i * s0A;
        const int baseC = offC + i * s0C;

        for (int kk = 0; kk < colA; kk += blocoColA) {
            const int fimK = (kk + blocoColA < colA) ? kk + blocoColA : colA;

            for (int jj = 0; jj < colB; jj += blocoColB) {
                const int fimJ = (jj + blocoColB < colB) ? jj + blocoColB : colB;

                for (int k = kk; k < fimK; k++) {
                    const double valA = A[baseA + k * s1A];
                    const int baseB = offB + k * s0B;

                    #pragma omp simd
                    for (int j = jj; j < fimJ; j++) {
                        C[baseC + j * s1C] += valA * B[baseB + j * s1B];
                    }
                }
            }
        }
    }

    (*env)->ReleasePrimitiveArrayCritical(env, aArr, A, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, bArr, B, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, cArr, C, 0);
}