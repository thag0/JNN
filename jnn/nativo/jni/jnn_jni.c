#include <jni.h>

#include "dispatcher.h"
#include "matmul.h"
#include "conv2d.h"
#include "maxpool.h"

JNIEXPORT void JNICALL
Java_jnn_nativo_JNNNative_matmul(
    JNIEnv* env, jclass cls,
    jdoubleArray A_arr, jint off_a, jint std_a_0, jint std_a_1,
    jdoubleArray B_arr, jint off_b, jint std_b_0, jint std_b_1,
    jdoubleArray C_arr, jint off_c, jint std_c_0, jint std_c_1,
    jint lin_a, jint col_a, jint col_b
) {
    (void) cls;

    matmul_params_t p;

    p.A = (*env)->GetPrimitiveArrayCritical(env, A_arr, NULL);
    p.B = (*env)->GetPrimitiveArrayCritical(env, B_arr, NULL);
    p.DST = (*env)->GetPrimitiveArrayCritical(env, C_arr, NULL);

    p.off_a = off_a;
    p.off_b = off_b;
    p.off_dst = off_c;

    p.std_a_0 = std_a_0;
    p.std_a_1 = std_a_1;
    p.std_b_0 = std_b_0;
    p.std_b_1 = std_b_1;
    p.std_c_0 = std_c_0;
    p.std_c_1 = std_c_1;

    p.lin_a = lin_a;
    p.col_a = col_a;
    p.col_b = col_b;

    jnn_matmul_dispatcher(&p);

    (*env)->ReleasePrimitiveArrayCritical(env, A_arr, p.A, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, B_arr, p.B, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, C_arr, p.DST, 0);
}

JNIEXPORT void JNICALL
Java_jnn_nativo_JNNNative_conv2dForward(
    JNIEnv* env, jclass cls,
    jdoubleArray X_arr, jint off_x,
    jdoubleArray K_arr, jint off_k,
    jdoubleArray B_arr, jint off_b, jboolean temBias,
    jdoubleArray DST_arr, jint off_dst,
    jint lotes, jint canais, jint filtros,
    jint alt_x, jint larg_x,
    jint alt_k, jint larg_k
) {
    (void) cls;

    double* restrict X = (*env)->GetPrimitiveArrayCritical(env, X_arr, NULL);
    double* restrict K = (*env)->GetPrimitiveArrayCritical(env, K_arr, NULL);
    double* restrict DST = (*env)->GetPrimitiveArrayCritical(env, DST_arr, NULL);
    double* restrict B = temBias ? (*env)->GetPrimitiveArrayCritical(env, B_arr, NULL) : NULL;

    conv2d_fwd_params_t p = {
        .X = X,
        .K = K,
        .B = B,
        .DST = DST,

        .off_x = off_x,
        .off_k = off_k,
        .off_b = off_b,
        .off_dst = off_dst,

        .lotes   = lotes,
        .canais  = canais,
        .filtros = filtros,

        .alt_x  = alt_x,
        .larg_x = larg_x,
        .alt_k  = alt_k,
        .larg_k = larg_k,

        .temBias = temBias
    };

    jnn_conv2d_fw_dispatcher(&p);

    (*env)->ReleasePrimitiveArrayCritical(env, X_arr, X, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, K_arr, K, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, DST_arr, DST, 0);

    if (temBias) {
        (*env)->ReleasePrimitiveArrayCritical(env, B_arr, B, JNI_ABORT);
    }
}

JNIEXPORT void JNICALL
Java_jnn_nativo_JNNNative_conv2dBackward(
    JNIEnv* env, jclass cls,
    jdoubleArray X_arr,  jint off_x,
    jdoubleArray K_arr,  jint off_k,
    jdoubleArray GS_arr, jint off_gs,
    jdoubleArray GK_arr, jint off_gk,
    jdoubleArray GB_arr, jint off_gb, jboolean temBias,
    jdoubleArray GE_arr, jint off_ge,
    jint lotes, jint canais, jint filtros,
    jint alt_x, jint larg_x,
    jint alt_k, jint larg_k
) {
    (void) cls;

    double* X  = (*env)->GetPrimitiveArrayCritical(env, X_arr, NULL);
    double* K  = (*env)->GetPrimitiveArrayCritical(env, K_arr, NULL);
    double* GS = (*env)->GetPrimitiveArrayCritical(env, GS_arr, NULL);
    double* GK = (*env)->GetPrimitiveArrayCritical(env, GK_arr, NULL);
    double* GE = (*env)->GetPrimitiveArrayCritical(env, GE_arr, NULL);
    double* GB = temBias ? (*env)->GetPrimitiveArrayCritical(env, GB_arr, NULL) : NULL;

    conv2d_bwd_params_t p = {
        .X  = X,
        .K  = K,
        .GS = GS,
        .GK = GK,
        .GE = GE,
        .GB = GB,

        .off_x  = off_x,
        .off_k  = off_k,
        .off_gs = off_gs,
        .off_gk = off_gk,
        .off_ge = off_ge,
        .off_gb = off_gb,

        .lotes   = lotes,
        .canais  = canais,
        .filtros = filtros,

        .alt_x  = alt_x,
        .larg_x = larg_x,
        .alt_k  = alt_k,
        .larg_k = larg_k,

        .temBias = temBias
    };

    jnn_conv2d_bw_dispatcher(&p);

    (*env)->ReleasePrimitiveArrayCritical(env, X_arr,  X,  JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, K_arr,  K,  JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, GS_arr, GS, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, GK_arr, GK, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, GE_arr, GE, 0);

    if (temBias) {
        (*env)->ReleasePrimitiveArrayCritical(env, GB_arr, GB, 0);
    }
}

JNIEXPORT void JNICALL
Java_jnn_nativo_JNNNative_maxPool2dForward(
    JNIEnv* env, jclass cls,
    jdoubleArray x_arr, jint off_x,
    jdoubleArray y_arr, jint off_y,
    jint lotes, jint canais,
    jint alt_x, jint larg_x,
    jint alt_pool, jint larg_pool,
    jint alt_std, jint larg_std
) {
    (void) cls;

    double* X = (*env)->GetPrimitiveArrayCritical(env, x_arr, NULL);
    double* Y = (*env)->GetPrimitiveArrayCritical(env, y_arr, NULL);

    maxpool2d_fwd_params_t p = {
        .X = X,
        .Y = Y,
        .off_x = off_x,
        .off_y = off_y,
        .lotes = lotes,
        .canais = canais,
        .alt_x = alt_x,
        .larg_x = larg_x,
        .alt_pool = alt_pool,
        .larg_pool = larg_pool,
        .alt_std = alt_std,
        .larg_std = larg_std
    };

    jnn_maxpool2d_fw_dispatcher(&p);

    (*env)->ReleasePrimitiveArrayCritical(env, x_arr, X, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, y_arr, Y, 0);
}

JNIEXPORT void JNICALL
Java_jnn_nativo_JNNNative_maxPool2dBackward(
    JNIEnv* env, jclass cls,
    jdoubleArray x_arr, jint off_x,
    jdoubleArray gs_arr, jint off_gs,
    jdoubleArray ge_arr, jint off_ge,
    jint lotes, jint canais,
    jint alt_x, jint larg_x,
    jint alt_gs, jint larg_gs,
    jint alt_pool, jint larg_pool,
    jint alt_std, jint larg_std
) {

    double* X = (*env)->GetPrimitiveArrayCritical(env, x_arr, NULL);
    double* GS = (*env)->GetPrimitiveArrayCritical(env, gs_arr, NULL);
    double* GE = (*env)->GetPrimitiveArrayCritical(env, ge_arr, NULL);

    maxpool2d_bwd_params_t p = {
        .X = X,
        .GS = GS,
        .GE = GE,
        .off_x = off_x,
        .off_gs = off_gs,
        .off_ge = off_ge,
        .lotes = lotes,
        .canais = canais,
        .alt_x = alt_x,
        .larg_x = larg_x,
        .alt_pool = alt_pool,
        .larg_pool = larg_pool,
        .alt_std = alt_std,
        .larg_std = larg_std,
    };

    jnn_maxpool2d_bw_dispatcher(&p);

    (*env)->ReleasePrimitiveArrayCritical(env, x_arr, X, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, gs_arr, GS, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, ge_arr, GE, 0);
}