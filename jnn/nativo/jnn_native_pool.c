#include "jnn_native_common.h"

#define MIN_DOUBLE_VAL -1e300

static inline void max_pool_2d(
    const double* restrict X, int offX,
    double* restrict Y, int offY,
    int canais,
    int x_h, int x_w,
    int pool_h, int pool_w,
    int std_h, int std_w
) {
    const int y_h = (x_h - pool_h) / std_h + 1;
    const int y_w = (x_w - pool_w) / std_w + 1;

    const int area_x = x_h * x_w;
    const int area_y = y_h * y_w;

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < canais; c++) {
        const int off_x_c = offX + c * area_x;
        const int off_y_c = offY + c * area_y;

        for (int i = 0; i < y_h; i++) {
            const int base_x_h = off_x_c + i * std_h * x_w;
            const int base_y_h = off_y_c + i * y_w;

            for (int j = 0; j < y_w; j++) {
                double max = MIN_DOUBLE_VAL;
                const int base_x_w = base_x_h + j * std_w;

                for (int ph = 0; ph < pool_h; ph++) {
                    const int lin_x = base_x_w + ph * x_w;
                    for (int pw = 0; pw < pool_w; pw++) {
                        double v = X[lin_x + pw];
                        if (v > max) max = v;
                    }
                }

                Y[base_y_h + j] = max;
            }
        }
    }
}

static inline void max_pool_2d_lotes(
    const double* restrict X, int offX,
    double* restrict Y, int offY,
    int lotes, int canais,
    int x_h, int x_w,
    int pool_h, int pool_w,
    int std_h, int std_w
) {
    const int y_h = (x_h - pool_h) / std_h + 1;
    const int y_w = (x_w - pool_w) / std_w + 1;

    const int area_x = x_h * x_w;
    const int area_y = y_h * y_w;

    const int std_lote_x = canais * area_x;
    const int std_lote_y = canais * area_y;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < lotes; b++) {
        for (int c = 0; c < canais; c++) {

            const int off_x_bc = offX + b * std_lote_x + c * area_x;
            const int off_y_bc = offY + b * std_lote_y + c * area_y;

            for (int i = 0; i < y_h; i++) {
                const int base_x_h = off_x_bc + i * std_h * x_w;
                const int base_y_h = off_y_bc + i * y_w;

                for (int j = 0; j < y_w; j++) {
                    double max = MIN_DOUBLE_VAL;
                    const int base_x_w = base_x_h + j * std_w;

                    for (int ph = 0; ph < pool_h; ph++) {
                        const int lin_x = base_x_w + ph * x_w;
                        for (int pw = 0; pw < pool_w; pw++) {
                            double v = X[lin_x + pw];
                            if (v > max) max = v;
                        }
                    }

                    Y[base_y_h + j] = max;
                }
            }
        }
    }
}

JNIEXPORT void JNICALL
Java_jnn_nativo_JNNNative_maxPool2dForward(
    JNIEnv* env, jclass cls,
    jdoubleArray XArr, jint offX,
    jdoubleArray YArr, jint offY,
    jint canais,
    jint altX, jint largX,
    jint poolH, jint poolW,
    jint strideH, jint strideW
) {
    (void) cls;

    double* restrict X = (*env)->GetPrimitiveArrayCritical(env, XArr, NULL);
    double* restrict Y = (*env)->GetPrimitiveArrayCritical(env, YArr, NULL);

    max_pool_2d(
        X, offX,
        Y, offY,
        canais,
        altX, largX,
        poolH, poolW,
        strideH, strideW
    );

    (*env)->ReleasePrimitiveArrayCritical(env, XArr, X, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, YArr, Y, 0);
}

JNIEXPORT void JNICALL
Java_jnn_nativo_JNNNative_maxPool2dForwardLotes(
    JNIEnv* env, jclass cls,
    jdoubleArray XArr, jint offX,
    jdoubleArray YArr, jint offY,
    jint lotes,
    jint canais,
    jint altX, jint largX,
    jint poolH, jint poolW,
    jint strideH, jint strideW
) {
    (void) cls;

    double* restrict X = (*env)->GetPrimitiveArrayCritical(env, XArr, NULL);
    double* restrict Y = (*env)->GetPrimitiveArrayCritical(env, YArr, NULL);

    max_pool_2d_lotes(
        X, offX,
        Y, offY,
        lotes,
        canais,
        altX, largX,
        poolH, poolW,
        strideH, strideW
    );

    (*env)->ReleasePrimitiveArrayCritical(env, XArr, X, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, YArr, Y, 0);
}

JNIEXPORT void JNICALL
Java_jnn_nativo_JNNNative_maxPool2dBackward(
    JNIEnv* env, jclass cls,

    jdoubleArray entradaArr, jint offE,
    jint canais, jint altE, jint largE,

    jdoubleArray gradSArr, jint offGS,
    jint altGS, jint largGS,

    jdoubleArray gradEArr, jint offGE,

    jint filtroH, jint filtroW,
    jint strideH, jint strideW
) {

    double* restrict entrada = (*env)->GetPrimitiveArrayCritical(env, entradaArr, NULL);
    double* restrict gradS   = (*env)->GetPrimitiveArrayCritical(env, gradSArr, NULL);
    double* restrict gradE   = (*env)->GetPrimitiveArrayCritical(env, gradEArr, NULL);

    const int area_x  = altE  * largE;
    const int area_gs = altGS * largGS;

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < canais; c++) {

        const int base_e  = offE  + c * area_x;
        const int base_gs = offGS + c * area_gs;
        const int base_ge = offGE + c * area_x;

        for (int i = 0; i < altGS; i++) {
            const int lin_ini_i = i * strideH;
            const int lin_fim = (lin_ini_i + filtroH < altE) ? (lin_ini_i + filtroH) : altE;

            for (int j = 0; j < largGS; j++) {
                const int col_ini_i = j * strideW;
                const int col_fim = (col_ini_i + filtroW < largE) ? (col_ini_i + filtroW) : largE;

                double val_max = MIN_DOUBLE_VAL;
                int lin_max = lin_ini_i;
                int col_max = col_ini_i;

                for (int y = lin_ini_i; y < lin_fim; y++) {
                    const int linha = base_e + y * largE;
                    for (int x = col_ini_i; x < col_fim; x++) {
                        double v = entrada[linha + x];
                        if (v > val_max) {
                            val_max = v;
                            lin_max = y;
                            col_max = x;
                        }
                    }
                }

                gradE[base_ge + lin_max * largE + col_max] += gradS[base_gs + i * largGS + j];
            }
        }
    }

    (*env)->ReleasePrimitiveArrayCritical(env, entradaArr, entrada, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, gradSArr,   gradS,   0);
    (*env)->ReleasePrimitiveArrayCritical(env, gradEArr,   gradE,   0);
}

JNIEXPORT void JNICALL
Java_jnn_nativo_JNNNative_maxPool2dBackwardLotes(
    JNIEnv* env, jclass cls,

    jdoubleArray entradaArr,
    jdoubleArray gradSArr,
    jdoubleArray gradEArr,

    jint lotes, jint canais,

    jint altE, jint largE,
    jint altGS, jint largGS,

    jint filtroH, jint filtroW,
    jint strideH, jint strideW
) {
    double* restrict X = (*env)->GetPrimitiveArrayCritical(env, entradaArr, NULL);
    double* restrict GS = (*env)->GetPrimitiveArrayCritical(env, gradSArr, NULL);
    double* restrict GE = (*env)->GetPrimitiveArrayCritical(env, gradEArr, NULL);

    const int area_e  = altE  * largE;
    const int area_gs = altGS * largGS;

    const int bloco_e  = canais * area_e;
    const int bloco_gs = canais * area_gs;

    #pragma omp parallel for schedule(static)
    for (int bc = 0; bc < lotes * canais; bc++) {

        const int b = bc / canais;
        const int c = bc % canais;

        const int base_e  = b * bloco_e  + c * area_e;
        const int base_gs = b * bloco_gs + c * area_gs;
        const int base_ge = base_e;

        for (int i = 0; i < altGS; i++) {
            const int lin_ini = i * strideH;
            const int lin_fim = (lin_ini + filtroH < altE) ? (lin_ini + filtroH) : altE;

            for (int j = 0; j < largGS; j++) {
                const int col_ini = j * strideW;
                const int col_fim = (col_ini + filtroW < largE) ? (col_ini + filtroW) : largE;

                double val_max = MIN_DOUBLE_VAL;
                int lin_max = lin_ini;
                int col_max = col_ini;

                for (int y = lin_ini; y < lin_fim; y++) {
                    const int linha = base_e + y * largE;
                    for (int x = col_ini; x < col_fim; x++) {
                        double v = X[linha + x];
                        if (v > val_max) {
                            val_max = v;
                            lin_max = y;
                            col_max = x;
                        }
                    }
                }

                GE[base_ge + lin_max * largE + col_max] += GS[base_gs + i * largGS + j];
            }
        }
    }

    (*env)->ReleasePrimitiveArrayCritical(env, entradaArr, X, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, gradSArr,   GS,   0);
    (*env)->ReleasePrimitiveArrayCritical(env, gradEArr,   GE,   0);
}