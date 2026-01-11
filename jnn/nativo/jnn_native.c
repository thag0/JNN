#include <jni.h>
#include <stdio.h>
#include <omp.h>

static inline int jnn_native_num_threads() {
    int p = omp_get_num_procs();
    return p > 1 ? p / 2 : 1;
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    omp_set_num_threads(jnn_native_num_threads());
    return JNI_VERSION_1_8;
}

JNIEXPORT void JNICALL
Java_jnn_nativo_JNNNative_setThreads(
    JNIEnv* env, jclass cls, jint n
) {
    (void) env;
    (void) cls;

    if (n < 1) {
        n = 1;
    }

    omp_set_num_threads((int)n);
}

JNIEXPORT void JNICALL
Java_jnn_nativo_JNNNative_matmul(
    JNIEnv* env, jclass cls,
    jdoubleArray aArr, jint offA, jint s0A, jint s1A,
    jdoubleArray bArr, jint offB, jint s0B, jint s1B,
    jdoubleArray cArr, jint offC, jint s0C, jint s1C,
    jint linA, jint colA, jint colB
) {
    (void) cls;
    
    jboolean isCopyA, isCopyB, isCopyC;

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

JNIEXPORT void JNICALL
Java_jnn_nativo_JNNNative_conv2dForward(
    JNIEnv* env, jclass cls,
    jdoubleArray XArr, jint offX,
    jdoubleArray KArr, jint offK,
    jdoubleArray BArr, jint offB, jboolean temBias,
    jdoubleArray YArr, jint offY,
    jint lotes, jint canais, jint filtros,
    jint altX, jint largX,
    jint altK, jint largK
) {
    (void) cls;

    double* restrict X = (*env)->GetPrimitiveArrayCritical(env, XArr, NULL);
    double* restrict K = (*env)->GetPrimitiveArrayCritical(env, KArr, NULL);
    double* restrict Y = (*env)->GetPrimitiveArrayCritical(env, YArr, NULL);
    double* restrict B = temBias ? (*env)->GetPrimitiveArrayCritical(env, BArr, NULL) : NULL;

    const int altS = altX - altK + 1;
    const int largS = largX - largK + 1;

    const int areaX = altX * largX;
    const int areaK = altK * largK;
    const int areaS = altS * largS;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < lotes; b++) {
        for (int f = 0; f < filtros; f++) {
            const int offXb  = offX + b * canais * areaX;
            const int offYb  = offY + b * filtros * areaS;
            const int offYbf = offYb + f * areaS;
            const int offKf  = offK + f * canais * areaK;
            const double bias = temBias ? B[offB + f] : 0.0;

            for (int i = 0; i < altS; i++) {
                double* restrict linDst = Y + offYbf + i * largS;

                for (int j = 0; j < largS; j++) {
                    linDst[j] = bias;
                }
            }

            for (int c = 0; c < canais; c++) {
                const int offXc = offXb + c * areaX;
                const int offKc = offKf + c * areaK;

                for (int kh = 0; kh < altK; kh++) {
                    const int linK = offKc + kh * largK;

                    for (int kw = 0; kw < largK; kw++) {
                        const double kval = K[linK + kw];
                        const int xBase = offXc + kh * largX + kw;

                        for (int i = 0; i < altS; i++) {
                            double* restrict linDst = Y + offYbf + i * largS;
                            const double* restrict linX = X + xBase + i * largX;

                            #pragma omp simd
                            for (int j = 0; j < largS; j++) {
                                linDst[j] += linX[j] * kval;
                            }
                        }
                    }
                }
            }
        }
    }

    (*env)->ReleasePrimitiveArrayCritical(env, XArr, X, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, KArr, K, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, YArr, Y, 0);

    if (temBias) {
        (*env)->ReleasePrimitiveArrayCritical(env, BArr, B, JNI_ABORT);
    }
}

static inline void corr2d(
    const double* restrict dataX, int offX,
    const double* restrict dataK, int offK,
    double* restrict dataDst, int offDst,
    int altX, int largX,
    int altK, int largK
) {
    const int altS = altX - altK + 1;
    const int largS = largX - largK + 1;

    for (int i = 0; i < altS; i++) {
        int baseS = offDst + i * largS;
        int baseX = offX + i * largX;

        for (int j = 0; j < largS; j++) {
            double soma = 0.0;
            int colBaseX = baseX + j;

            for (int kh = 0; kh < altK; kh++) {
                int linX = colBaseX + kh * largX;
                int linK = offK + kh * largK;

                for (int kw = 0; kw < largK; kw++) {
                    soma += dataX[linX + kw] * dataK[linK + kw];
                }
            }

            dataDst[baseS + j] += soma;
        }
    }
}

static inline void conv2d_full(
    const double* restrict dataX, int offX,
    const double* restrict dataK, int offK,
    double* restrict dataDst, int offDst,
    int altX, int largX,
    int altK, int largK
) {
    const int altS = altX + altK - 1;
    const int largS = largX + largK - 1;

    for (int i = 0; i < altS; i++) {
        const int baseS = offDst + i * largS;

        for (int j = 0; j < largS; j++) {
            double soma = 0.0;

            for (int kh = 0; kh < altK; kh++) {
                int linX = i - kh;
                if (linX < 0 || linX >= altX) continue;

                int baseX = offX + linX * largX;
                int baseK = offK + kh * largK;

                for (int kw = 0; kw < largK; kw++) {
                    int colX = j - kw;
                    if (colX < 0 || colX >= largX) continue;

                    soma += dataK[baseK + kw] * dataX[baseX + colX];
                }
            }

            dataDst[baseS + j] += soma;
        }
    }
}

JNIEXPORT void JNICALL
Java_jnn_nativo_JNNNative_conv2dBackward(
    JNIEnv* env, jclass cls,
    jdoubleArray XArr,  jint offXbase,
    jdoubleArray KArr,  jint offKbase,
    jdoubleArray gradSArr, jint offGSbase,
    jdoubleArray gradKArr, jint offGKbase,
    jdoubleArray gradBArr, jint offGBbase, jboolean temBias,
    jdoubleArray gradEArr, jint offGEbase,
    jint lotes, jint canais, jint filtros,
    jint altX, jint largX,
    jint altK, jint largK
) {
    (void) cls;
    
    double *restrict X  = (*env)->GetPrimitiveArrayCritical(env, XArr, NULL);
    double* restrict K  = (*env)->GetPrimitiveArrayCritical(env, KArr, NULL);
    double* restrict GS = (*env)->GetPrimitiveArrayCritical(env, gradSArr, NULL);
    double* restrict GK = (*env)->GetPrimitiveArrayCritical(env, gradKArr, NULL);
    double* restrict GE = (*env)->GetPrimitiveArrayCritical(env, gradEArr, NULL);
    double* restrict GB = temBias ? (*env)->GetPrimitiveArrayCritical(env, gradBArr, NULL) : NULL;

    const int altS = altX - altK + 1;
    const int largS = largX - largK + 1;

    const int areaX = altX * largX;
    const int areaK = altK * largK;
    const int areaGS = altS * largS;

    #pragma omp parallel for schedule(static) 
    for (int f = 0; f < filtros; f++) {
        const int filtro = f;

        const int offKf = offKbase + filtro * canais * areaK;
        double somaBiasLocal = 0;

        for (int l = 0; l < lotes; l++) {
            const int offGS = offGSbase + (l * filtros + f) * areaGS;
            for (int c = 0; c < canais; c++) {
                int offX = offXbase + (l * canais + c) * areaX;
                int offGK = offKf + c * areaK;
                corr2d(
                    X, offX,
                    GS, offGS,
                    GK, offGK,
                    altX, largX,
                    altS, largS
                );
            }

            if (temBias) {
                for (int i = 0; i < areaGS; i++) {
                    somaBiasLocal += GS[offGS + i];
                }
            }
        }

        if (temBias) {
            GB[offGBbase + f] += somaBiasLocal;
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int l = 0; l < lotes; l++) {
        for (int c = 0; c < canais; c++) {
            int offGE = offGEbase + (l * canais + c) * areaX;
            for (int f = 0; f < filtros; f++) {
                int offGS = offGSbase + (l * filtros + f) * areaGS;
                int offK  = offKbase  + (f * canais + c) * areaK;
                conv2d_full(
                    GS, offGS,
                    K,  offK,
                    GE, offGE,
                    altS, largS,
                    altK, largK
                );
            }
        }
    }

    (*env)->ReleasePrimitiveArrayCritical(env, XArr, X, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, KArr, K, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, gradSArr, GS, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, gradKArr, GK, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, gradEArr, GE, 0);

    if (temBias) {
        (*env)->ReleasePrimitiveArrayCritical(env, gradBArr, GB, 0);
    }
}