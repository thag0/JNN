#include <jni.h>
#include <omp.h>

#include "dispatcher.h"
#include "matmul.h"
#include "conv2d.h"
#include "maxpool.h"

static inline int jnn_native_num_threads() {
    int p = omp_get_num_procs();
    return p > 1 ? p / 2 : 1;
}

JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM* vm, void* reserved) {
    (void) vm;
    (void) reserved;

    omp_set_num_threads(jnn_native_num_threads());
    return JNI_VERSION_1_8;
}

JNIEXPORT void JNICALL
Java_jnn_nativo_JNNNative_setThreads(JNIEnv* env, jclass cls, jint n) {
    (void) env;
    (void) cls;

    if (n < 1) n = 1;
    omp_set_num_threads((int)n);
}

JNIEXPORT void JNICALL 
Java_jnn_nativo_JNNNative_setBackend(JNIEnv * env, jclass cls, jint backend) {
    jnn_set_backend(backend);
}

JNIEXPORT void JNICALL
Java_jnn_nativo_JNNNative_matmul(
    JNIEnv* env, jclass cls,
    jdoubleArray A_arr, jint off_a, jint std_a_0, jint std_a_1,
    jdoubleArray B_arr, jint off_b, jint std_b_0, jint std_b_1,
    jdoubleArray C_arr, jint off_c, jint std_c_0, jint std_c_1,
    jint lin_a, jint col_a, jint col_b
) {
    (void) cls;

    matmul_params_t p = {
        .A = (*env)->GetPrimitiveArrayCritical(env, A_arr, NULL),
        .B = (*env)->GetPrimitiveArrayCritical(env, B_arr, NULL),
        .DST = (*env)->GetPrimitiveArrayCritical(env, C_arr, NULL),
    
        .off_a = off_a,
        .off_b = off_b,
        .off_dst = off_c,
    
        .std_a_0 = std_a_0,
        .std_a_1 = std_a_1,
        .std_b_0 = std_b_0,
        .std_b_1 = std_b_1,
        .std_c_0 = std_c_0,
        .std_c_1 = std_c_1,
    
        .lin_a = lin_a,
        .col_a = col_a,
        .col_b = col_b
    };


    jnn_matmul_dispatcher(&p);

    (*env)->ReleasePrimitiveArrayCritical(env, A_arr, p.A, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, B_arr, p.B, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, C_arr, p.DST, 0);
}

JNIEXPORT void JNICALL
Java_jnn_nativo_JNNNative_conv2dForward(
    JNIEnv* env, jclass cls,
    jdoubleArray X_arr,
    jdoubleArray K_arr,
    jdoubleArray B_arr,
    jboolean temBias,
    jdoubleArray DST_arr,
    jint lotes, 
    jint canais, 
    jint filtros,
    jint alt_x, 
    jint larg_x,
    jint alt_k, 
    jint larg_k
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
    jdoubleArray X_arr,
    jdoubleArray K_arr,
    jdoubleArray GS_arr,
    jdoubleArray GK_arr,
    jdoubleArray GB_arr, 
    jboolean temBias,
    jdoubleArray GE_arr,
    jint lotes, 
    jint canais, 
    jint filtros,
    jint alt_x, 
    jint larg_x,
    jint alt_k, 
    jint larg_k
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
    jdoubleArray x_arr,
    jdoubleArray y_arr,
    jint lotes, 
    jint canais,
    jint alt_x, 
    jint larg_x,
    jint alt_pool, 
    jint larg_pool,
    jint alt_std, 
    jint larg_std
) {
    (void) cls;

    double* X = (*env)->GetPrimitiveArrayCritical(env, x_arr, NULL);
    double* Y = (*env)->GetPrimitiveArrayCritical(env, y_arr, NULL);

    maxpool2d_fwd_params_t p = {
        .X = X,
        .Y = Y,
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
    jdoubleArray x_arr,
    jdoubleArray gs_arr,
    jdoubleArray ge_arr,
    jint lotes, 
    jint canais,
    jint alt_x, 
    jint larg_x,
    jint alt_gs, 
    jint larg_gs,
    jint alt_pool, 
    jint larg_pool,
    jint alt_std, 
    jint larg_std
) {
    double* X = (*env)->GetPrimitiveArrayCritical(env, x_arr, NULL);
    double* GS = (*env)->GetPrimitiveArrayCritical(env, gs_arr, NULL);
    double* GE = (*env)->GetPrimitiveArrayCritical(env, ge_arr, NULL);

    maxpool2d_bwd_params_t p = {
        .X = X,
        .GS = GS,
        .GE = GE,
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