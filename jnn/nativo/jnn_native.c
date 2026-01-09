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
    jint M, jint K, jint N
) {
    (void) cls;
    
    jboolean isCopyA, isCopyB, isCopyC;

    double* restrict A = (*env)->GetPrimitiveArrayCritical(env, aArr, NULL);
    double* restrict B = (*env)->GetPrimitiveArrayCritical(env, bArr, NULL);
    double* restrict C = (*env)->GetPrimitiveArrayCritical(env, cArr, NULL);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        int baseA = offA + i * s0A;
        int baseC = offC + i * s0C;

        for (int k = 0; k < K; k++) {
            double valA = A[baseA + k * s1A];
            int baseB = offB + k * s0B;

            for (int j = 0; j < N; j++) {
                C[baseC + j * s1C] += valA * B[baseB + j * s1B];
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
    jdoubleArray BArr, jint offB, jboolean hasBias,
    jdoubleArray YArr, jint offY,
    jint BATCH, jint CIN, jint COUT,
    jint H, jint W,
    jint kH, jint kW
) {
    (void) cls;

    double* restrict X = (*env)->GetPrimitiveArrayCritical(env, XArr, NULL);
    double* restrict K = (*env)->GetPrimitiveArrayCritical(env, KArr, NULL);
    double* restrict Y = (*env)->GetPrimitiveArrayCritical(env, YArr, NULL);
    double* restrict B = hasBias
        ? (*env)->GetPrimitiveArrayCritical(env, BArr, NULL)
        : NULL;

    const int outH = H - kH + 1;
    const int outW = W - kW + 1;

    const int areaX = H * W;
    const int areaK = kH * kW;
    const int areaY = outH * outW;

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < BATCH; b++) {
        for (int f = 0; f < COUT; f++) {
            const int offXb  = offX + b * CIN * areaX;
            const int offYb  = offY + b * COUT * areaY;
            const int offYbf = offYb + f * areaY;
            const int offKf  = offK + f * CIN * areaK;
            const double bias = hasBias ? B[offB + f] : 0.0;

            for (int i = 0; i < outH; i++) {
                double* restrict yRow =
                    Y + offYbf + i * outW;

                for (int j = 0; j < outW; j++) {
                    yRow[j] = bias;
                }
            }

            for (int c = 0; c < CIN; c++) {
                const int offXc = offXb + c * areaX;
                const int offKc = offKf + c * areaK;

                for (int kh = 0; kh < kH; kh++) {
                    const int kRow = offKc + kh * kW;

                    for (int kw = 0; kw < kW; kw++) {
                        const double kval = K[kRow + kw];

                        const int xBase = offXc + kh * W + kw;

                        for (int i = 0; i < outH; i++) {
                            double* restrict yRow =
                                Y + offYbf + i * outW;

                            const double* restrict xRow =
                                X + xBase + i * W;

                            for (int j = 0; j < outW; j++) {
                                yRow[j] += xRow[j] * kval;
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

    if (hasBias) {
        (*env)->ReleasePrimitiveArrayCritical(env, BArr, B, JNI_ABORT);
    }
}

static inline void conv2d_full(
    const double* restrict dataX, int offX,   // gradS
    const double* restrict dataK, int offK,   // kernel
    double* restrict dataDst, int offDst,     // gradE
    int W, int H,                             // dimensões de gradS
    int kW, int kH
) {
    const int outH = H + kH - 1;
    const int outW = W + kW - 1;

    for (int i = 0; i < outH; i++) {
        const int baseOut = offDst + i * outW;

        for (int j = 0; j < outW; j++) {
            double sum = 0.0;

            for (int kh = 0; kh < kH; kh++) {
                int inRow = i - kh;
                if (inRow < 0 || inRow >= H) continue;

                int baseIn = offX + inRow * W;
                int baseK = offK + kh * kW;

                for (int kw = 0; kw < kW; kw++) {
                    int inCol = j - kw;
                    if (inCol < 0 || inCol >= W) continue;

                    sum += dataK[baseK + kw] * dataX[baseIn + inCol];
                }
            }

            dataDst[baseOut + j] += sum;
        }
    }
}

static inline void corr2d(
    const double* restrict dataX, int offX,   // entrada X
    const double* restrict dataK, int offK,   // gradS
    double* restrict dataDst, int offDst,     // gradK
    int W, int H,                             // dimensões da entrada X
    int kW, int kH
) {
    const int outH = H - kH + 1;
    const int outW = W - kW + 1;

    for (int i = 0; i < outH; i++) {
        int baseOut = offDst + i * outW;
        int baseIn = offX + i * W;

        for (int j = 0; j < outW; j++) {
            double sum = 0.0;
            int inColBase = baseIn + j;

            for (int kh = 0; kh < kH; kh++) {
                int inRow = inColBase + kh * W;
                int kRow = offK + kh * kW;

                for (int kw = 0; kw < kW; kw++) {
                    sum += dataX[inRow + kw] * dataK[kRow + kw];
                }
            }

            dataDst[baseOut + j] += sum;
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
    jdoubleArray gradBArr, jint offGBbase, jboolean hasBias,
    jdoubleArray gradEArr, jint offGEbase,
    jint lotes, jint canais, jint filtros,
    jint altX, jint largX,
    jint altK, jint largK
) {
    (void) cls;
    
    double* restrict X  = (*env)->GetPrimitiveArrayCritical(env, XArr, NULL);
    double* restrict K  = (*env)->GetPrimitiveArrayCritical(env, KArr, NULL);
    double* restrict GS = (*env)->GetPrimitiveArrayCritical(env, gradSArr, NULL);
    double* restrict GK = (*env)->GetPrimitiveArrayCritical(env, gradKArr, NULL);
    double* restrict GE = (*env)->GetPrimitiveArrayCritical(env, gradEArr, NULL);
    double* restrict GB = hasBias ? (*env)->GetPrimitiveArrayCritical(env, gradBArr, NULL) : NULL;

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
                int offGE = offGEbase + (l * canais + c) * areaX;
                int offK = offKf + (c * areaK);

                conv2d_full(
                    GS, offGS,
                    K, offK,
                    GE, offGE,
                    altS, largS,
                    altK, largK
                );
            }

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

            if (hasBias) {
                for (int i = 0; i < areaGS; i++) {
                    somaBiasLocal += GS[offGS + i];
                }
            }
        }

        if (hasBias) {
            GB[offGBbase + f] += somaBiasLocal;
        }
    }

    (*env)->ReleasePrimitiveArrayCritical(env, XArr, X, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, KArr, K, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, gradSArr, GS, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, gradKArr, GK, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, gradEArr, GE, 0);

    if (hasBias) {
        (*env)->ReleasePrimitiveArrayCritical(env, gradBArr, GB, 0);
    }
}