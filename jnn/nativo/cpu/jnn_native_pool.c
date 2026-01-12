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
    const double* restrict X, int off_x,
    double* restrict Y, int off_y,
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

            const int off_x_bc = off_x + b * std_lote_x + c * area_x;
            const int off_y_bc = off_y + b * std_lote_y + c * area_y;

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
    jdoubleArray x_arr, jint off_x,
    jdoubleArray y_arr, jint off_y,
    jint canais,
    jint alt_x, jint larg_x,
    jint alt_pool, jint larg_pool,
    jint alt_std, jint larg_std
) {
    (void) cls;

    double* restrict X = (*env)->GetPrimitiveArrayCritical(env, x_arr, NULL);
    double* restrict Y = (*env)->GetPrimitiveArrayCritical(env, y_arr, NULL);

    max_pool_2d(
        X, off_x,
        Y, off_y,
        canais,
        alt_x, larg_x,
        alt_pool, larg_pool,
        alt_std, larg_std
    );

    (*env)->ReleasePrimitiveArrayCritical(env, x_arr, X, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, y_arr, Y, 0);
}

JNIEXPORT void JNICALL
Java_jnn_nativo_JNNNative_maxPool2dForwardLotes(
    JNIEnv* env, jclass cls,
    jdoubleArray x_arr, jint off_x,
    jdoubleArray y_arr, jint off_y,
    jint lotes, jint canais,
    jint alt_x, jint larg_x,
    jint alt_pool, jint larg_pool,
    jint alt_std, jint larg_std
) {
    (void) cls;

    double* restrict X = (*env)->GetPrimitiveArrayCritical(env, x_arr, NULL);
    double* restrict Y = (*env)->GetPrimitiveArrayCritical(env, y_arr, NULL);

    max_pool_2d_lotes(
        X, off_x,
        Y, off_y,
        lotes,
        canais,
        alt_x, larg_x,
        alt_pool, larg_pool,
        alt_std, larg_std
    );

    (*env)->ReleasePrimitiveArrayCritical(env, x_arr, X, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, y_arr, Y, 0);
}

JNIEXPORT void JNICALL
Java_jnn_nativo_JNNNative_maxPool2dBackward(
    JNIEnv* env, jclass cls,
    jdoubleArray x_arr, jint off_x,
    jint canais, jint alt_x, jint larg_x,
    jdoubleArray gs_arr, jint off_gs,
    jint alt_gs, jint larg_gs,
    jdoubleArray ge_arr, jint off_ge,
    jint alt_pool, jint larg_pool,
    jint alt_std, jint larg_std
) {

    double* restrict X  = (*env)->GetPrimitiveArrayCritical(env, x_arr, NULL);
    double* restrict GS = (*env)->GetPrimitiveArrayCritical(env, gs_arr, NULL);
    double* restrict GE = (*env)->GetPrimitiveArrayCritical(env, ge_arr, NULL);

    const int area_x  = alt_x  * larg_x;
    const int area_gs = alt_gs * larg_gs;

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < canais; c++) {

        const int base_e  = off_x  + c * area_x;
        const int base_gs = off_gs + c * area_gs;
        const int base_ge = off_ge + c * area_x;

        for (int i = 0; i < alt_gs; i++) {
            const int lin_ini_i = i * alt_std;
            const int lin_fim = (lin_ini_i + alt_pool < alt_x) ? (lin_ini_i + alt_pool) : alt_x;

            for (int j = 0; j < larg_gs; j++) {
                const int col_ini_i = j * larg_std;
                const int col_fim = (col_ini_i + larg_pool < larg_x) ? (col_ini_i + larg_pool) : larg_x;

                double val_max = MIN_DOUBLE_VAL;
                int lin_max = lin_ini_i;
                int col_max = col_ini_i;

                for (int y = lin_ini_i; y < lin_fim; y++) {
                    const int linha = base_e + y * larg_x;
                    for (int x = col_ini_i; x < col_fim; x++) {
                        double v = X[linha + x];
                        if (v > val_max) {
                            val_max = v;
                            lin_max = y;
                            col_max = x;
                        }
                    }
                }

                GE[base_ge + lin_max * larg_x + col_max] += GS[base_gs + i * larg_gs + j];
            }
        }
    }

    (*env)->ReleasePrimitiveArrayCritical(env, x_arr, X, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, gs_arr, GS, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, ge_arr, GE, 0);
}

JNIEXPORT void JNICALL
Java_jnn_nativo_JNNNative_maxPool2dBackwardLotes(
    JNIEnv* env, jclass cls,
    jdoubleArray x_arr,
    jdoubleArray gs_arr,
    jdoubleArray ge_arr,
    jint lotes, jint canais,
    jint alt_x, jint larg_x,
    jint alt_gs, jint larg_gs,
    jint alt_pool, jint larg_pool,
    jint alt_std, jint larg_std
) {
    double* restrict X  = (*env)->GetPrimitiveArrayCritical(env, x_arr, NULL);
    double* restrict GS = (*env)->GetPrimitiveArrayCritical(env, gs_arr, NULL);
    double* restrict GE = (*env)->GetPrimitiveArrayCritical(env, ge_arr, NULL);

    const int area_x  = alt_x  * larg_x;
    const int area_gs = alt_gs * larg_gs;

    const int bloco_e  = canais * area_x;
    const int bloco_gs = canais * area_gs;

    #pragma omp parallel for schedule(static)
    for (int bc = 0; bc < lotes * canais; bc++) {
        const int b = bc / canais;
        const int c = bc % canais;
        const int base_e  = b * bloco_e  + c * area_x;
        const int base_gs = b * bloco_gs + c * area_gs;
        const int base_ge = base_e;

        for (int i = 0; i < alt_gs; i++) {
            const int lin_ini = i * alt_std;
            const int lin_fim = (lin_ini + alt_pool < alt_x) ? (lin_ini + alt_pool) : alt_x;

            for (int j = 0; j < larg_gs; j++) {
                const int col_ini = j * larg_std;
                const int col_fim = (col_ini + larg_pool < larg_x) ? (col_ini + larg_pool) : larg_x;
                double val_max = MIN_DOUBLE_VAL;
                int lin_max = lin_ini;
                int col_max = col_ini;

                for (int y = lin_ini; y < lin_fim; y++) {
                    const int linha = base_e + y * larg_x;
                    for (int x = col_ini; x < col_fim; x++) {
                        double v = X[linha + x];
                        if (v > val_max) {
                            val_max = v;
                            lin_max = y;
                            col_max = x;
                        }
                    }
                }

                GE[base_ge + lin_max * larg_x + col_max] += GS[base_gs + i * larg_gs + j];
            }
        }
    }

    (*env)->ReleasePrimitiveArrayCritical(env, x_arr, X, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, gs_arr, GS, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, ge_arr, GE, 0);
}