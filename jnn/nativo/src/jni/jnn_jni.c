#include <jni.h>
#include <omp.h>

#include "dispatcher.h"
#include "gemm.h"
#include "conv2d.h"
#include "maxpool.h"
#include "batchnorm2d.h"

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
Java_jnn_core_JNNnative_setThreads(JNIEnv* env, jclass cls, jint n) {
    (void) env;
    (void) cls;

    if (n < 1) n = 1;
    omp_set_num_threads((int)n);
}

JNIEXPORT void JNICALL 
Java_jnn_core_JNNnative_setBackend(JNIEnv * env, jclass cls, jint backend) {
    (void) env;
    (void) cls;

    jnn_set_backend(backend);
}

JNIEXPORT void JNICALL
Java_jnn_core_JNNnative_matmul(
    JNIEnv* env, jclass cls,
    jfloatArray A_arr, jint off_a, jint std_a_0, jint std_a_1,
    jfloatArray B_arr, jint off_b, jint std_b_0, jint std_b_1,
    jfloatArray C_arr, jint off_c, jint std_c_0, jint std_c_1,
    jint lin_a, jint col_a, jint col_b
) {
    (void) cls;

    gemm_params_t p = {
        .A = (*env)->GetPrimitiveArrayCritical(env, A_arr, NULL),
        .B = (*env)->GetPrimitiveArrayCritical(env, B_arr, NULL),
        .C = (*env)->GetPrimitiveArrayCritical(env, C_arr, NULL),
    
        .off_a = off_a,
        .off_b = off_b,
        .off_c = off_c,
    
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
    (*env)->ReleasePrimitiveArrayCritical(env, C_arr, p.C, 0);
}

JNIEXPORT void JNICALL
Java_jnn_core_JNNnative_conv2dForward(
    JNIEnv* env, jclass cls,
    jfloatArray X_arr,
    jfloatArray K_arr,
    jfloatArray B_arr,
    jboolean temBias,
    jfloatArray DST_arr,
    jint lotes, 
    jint canais, 
    jint filtros,
    jint alt_x, 
    jint larg_x,
    jint alt_k, 
    jint larg_k,
    jint alt_pad,
    jint larg_pad
) {
    (void) cls;

    float* restrict X = (*env)->GetPrimitiveArrayCritical(env, X_arr, NULL);
    float* restrict K = (*env)->GetPrimitiveArrayCritical(env, K_arr, NULL);
    float* restrict DST = (*env)->GetPrimitiveArrayCritical(env, DST_arr, NULL);
    float* restrict B = temBias ? (*env)->GetPrimitiveArrayCritical(env, B_arr, NULL) : NULL;

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
        .alt_pad = alt_pad,
        .larg_pad = larg_pad,

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
Java_jnn_core_JNNnative_conv2dBackward(
    JNIEnv* env, jclass cls,
    jfloatArray X_arr,
    jfloatArray K_arr,
    jfloatArray GS_arr,
    jfloatArray GK_arr,
    jfloatArray GB_arr, 
    jboolean temBias,
    jfloatArray GE_arr,
    jint lotes, 
    jint canais, 
    jint filtros,
    jint alt_x, 
    jint larg_x,
    jint alt_k, 
    jint larg_k,
    jint alt_pad,
    jint larg_pad
) {
    (void) cls;

    float* X  = (*env)->GetPrimitiveArrayCritical(env, X_arr, NULL);
    float* K  = (*env)->GetPrimitiveArrayCritical(env, K_arr, NULL);
    float* GS = (*env)->GetPrimitiveArrayCritical(env, GS_arr, NULL);
    float* GK = (*env)->GetPrimitiveArrayCritical(env, GK_arr, NULL);
    float* GE = (*env)->GetPrimitiveArrayCritical(env, GE_arr, NULL);
    float* GB = temBias ? (*env)->GetPrimitiveArrayCritical(env, GB_arr, NULL) : NULL;

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
        .alt_pad = alt_pad,
        .larg_pad = larg_pad,

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
Java_jnn_core_JNNnative_maxPool2dForward(
    JNIEnv* env, jclass cls,
    jfloatArray x_arr,
    jfloatArray y_arr,
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

    float* X = (*env)->GetPrimitiveArrayCritical(env, x_arr, NULL);
    float* Y = (*env)->GetPrimitiveArrayCritical(env, y_arr, NULL);

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
Java_jnn_core_JNNnative_maxPool2dBackward(
    JNIEnv* env, jclass cls,
    jfloatArray x_arr,
    jfloatArray gs_arr,
    jfloatArray ge_arr,
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
    (void) cls;

    float* X = (*env)->GetPrimitiveArrayCritical(env, x_arr, NULL);
    float* GS = (*env)->GetPrimitiveArrayCritical(env, gs_arr, NULL);
    float* GE = (*env)->GetPrimitiveArrayCritical(env, ge_arr, NULL);

    maxpool2d_bwd_params_t p = {
        .X = X,
        .GS = GS,
        .GE = GE,
        .lotes = lotes,
        .canais = canais,
        .alt_x = alt_x,
        .larg_x = larg_x,
        .alt_gs = alt_gs,
        .larg_gs = larg_gs,
        .alt_pool = alt_pool,
        .larg_pool = larg_pool,
        .alt_std = alt_std,
        .larg_std = larg_std,
    };

    jnn_maxpool2d_bw_dispatcher(&p);

    (*env)->ReleasePrimitiveArrayCritical(env, x_arr,   X, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, gs_arr, GS, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, ge_arr, GE, 0);
}

JNIEXPORT void JNICALL Java_jnn_core_JNNnative_batchNorm2DForward(
    JNIEnv *env,
    jclass cls, 
    jfloatArray x, 
    jfloatArray y, 
    jfloatArray gamma, 
    jfloatArray beta, 
    jfloatArray media_movel, 
    jfloatArray variancia_movel,
    jfloatArray media,
    jfloatArray var,
    jfloatArray x_norm,
    jint lotes, 
    jint canais, 
    jint alt_x, 
    jint larg_x, 
    jfloat momentum, 
    jfloat eps, 
    jboolean treinando
) {
    (void) cls;

    float* restrict X = (*env)->GetPrimitiveArrayCritical(env, x, NULL);
    float* restrict Y = (*env)->GetPrimitiveArrayCritical(env, y, NULL);
    float* restrict G = (*env)->GetPrimitiveArrayCritical(env, gamma, NULL);
    float* restrict B = (*env)->GetPrimitiveArrayCritical(env, beta, NULL);
    float* restrict MM = (*env)->GetPrimitiveArrayCritical(env, media_movel, NULL);
    float* restrict VM = (*env)->GetPrimitiveArrayCritical(env, variancia_movel, NULL);
    float* restrict MD = (*env)->GetPrimitiveArrayCritical(env, media, NULL);
    float* restrict VR = (*env)->GetPrimitiveArrayCritical(env, var, NULL);
    float* restrict XN = (*env)->GetPrimitiveArrayCritical(env, x_norm, NULL);

    bn2d_fwd_params_t p = {
        .x = X,
        .y = Y,
        .gamma = G,
        .beta = B,
        .media_movel = MM,
        .variancia_movel = VM,
        .media = MD,
        .var = VR,
        .x_norm = XN,
        
        .lotes = lotes,
        .canais = canais,
        .alt_x = alt_x,
        .larg_x = larg_x,
        
        .momentum = momentum,
        .eps = eps,

        .treinando = treinando
    };

    jnn_batchnorm2d_fw_dispatcher(&p);

    (*env)->ReleasePrimitiveArrayCritical(env, x, X, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, gamma, G, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, beta, B, JNI_ABORT);

    (*env)->ReleasePrimitiveArrayCritical(env, y, Y, 0);

    if (treinando) {
        (*env)->ReleasePrimitiveArrayCritical(env, media_movel, MM, 0);
        (*env)->ReleasePrimitiveArrayCritical(env, variancia_movel, VM, 0);
        (*env)->ReleasePrimitiveArrayCritical(env, media, MD, 0);
        (*env)->ReleasePrimitiveArrayCritical(env, var, VR, 0);
        (*env)->ReleasePrimitiveArrayCritical(env, x_norm, XN, 0);
        
    } else {
        (*env)->ReleasePrimitiveArrayCritical(env, media_movel, MM, JNI_ABORT);
        (*env)->ReleasePrimitiveArrayCritical(env, variancia_movel, VM, JNI_ABORT);
        (*env)->ReleasePrimitiveArrayCritical(env, media, MD, JNI_ABORT);
        (*env)->ReleasePrimitiveArrayCritical(env, var, VR, JNI_ABORT);
        (*env)->ReleasePrimitiveArrayCritical(env, x_norm, XN, JNI_ABORT);
    }
}


JNIEXPORT void JNICALL Java_jnn_core_JNNnative_batchNorm2DBackward(
    JNIEnv *env,
    jclass cls,
    jfloatArray x_norm,
    jfloatArray var,
    jfloatArray gamma,
    jfloatArray ge,
    jfloatArray gs,
    jfloatArray gg,
    jfloatArray gb,
    jint lotes,
    jint canais,
    jint alt_x,
    jint larg_x,
    jfloat eps
) {
    (void) cls;

    float* restrict XN = (*env)->GetPrimitiveArrayCritical(env, x_norm, NULL);
    float* restrict VAR = (*env)->GetPrimitiveArrayCritical(env, var, NULL);
    float* restrict GAMMA = (*env)->GetPrimitiveArrayCritical(env, gamma, NULL);
    float* restrict GE = (*env)->GetPrimitiveArrayCritical(env, ge, NULL);
    float* restrict GS = (*env)->GetPrimitiveArrayCritical(env, gs, NULL);
    float* restrict GG = (*env)->GetPrimitiveArrayCritical(env, gg, NULL);
    float* restrict GB = (*env)->GetPrimitiveArrayCritical(env, gb, NULL);

    bn2d_bwd_params_t p = {
        .x_norm = XN,
        .var = VAR,
        .gamma = GAMMA,
        .ge = GE,
        .gs = GS,
        .gg = GG,
        .gb = GB,

        .lotes = lotes,
        .canais = canais,
        .alt_x = alt_x,
        .larg_x = larg_x,
        
        .eps = eps,
    };

    jnn_batchnorm2d_bw_dispatcher(&p);

    (*env)->ReleasePrimitiveArrayCritical(env, x_norm, XN, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, var, VAR, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, gamma, GAMMA, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, gs, GS, JNI_ABORT);
    
    (*env)->ReleasePrimitiveArrayCritical(env, ge, GE, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, gg, GG, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, gb, GB, 0);

}

JNIEXPORT void JNICALL Java_jnn_core_JNNnative_relu(
    JNIEnv *env, 
    jclass cls, 
    jfloatArray src, 
    jfloatArray dst, 
    jint tam
) {
    (void) cls;
    
    float* restrict X = (*env)->GetPrimitiveArrayCritical(env, src, NULL);
    float* restrict Y = (*env)->GetPrimitiveArrayCritical(env, dst, NULL);

    jnn_relu(X, Y, (size_t) tam);

    (*env)->ReleasePrimitiveArrayCritical(env, src, (void*)X, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, dst, Y, 0);
}

JNIEXPORT void JNICALL Java_jnn_core_JNNnative_relud(
    JNIEnv *env, 
    jclass cls, 
    jfloatArray x,
    jfloatArray g, 
    jfloatArray dst, 
    jint tam
) { 
    (void) cls;

    float* restrict X = (*env)->GetPrimitiveArrayCritical(env, x, NULL);
    float* restrict G = (*env)->GetPrimitiveArrayCritical(env, g, NULL);
    float* restrict DST = (*env)->GetPrimitiveArrayCritical(env, dst, NULL);

    jnn_relu_d(X, G, DST, (size_t) tam);

    (*env)->ReleasePrimitiveArrayCritical(env, x, (void*)X, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, g, (void*)G, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, dst, DST, 0);
}


JNIEXPORT void JNICALL Java_jnn_core_JNNnative_sigmoid(
    JNIEnv *env, 
    jclass cls, 
    jfloatArray src, 
    jfloatArray dst,
    jint tam
) {
    (void) cls;

    float* restrict X = (*env)->GetPrimitiveArrayCritical(env, src, NULL);
    float* restrict Y = (*env)->GetPrimitiveArrayCritical(env, dst, NULL);

    sigmoid(X, Y, (size_t) tam);

    (*env)->ReleasePrimitiveArrayCritical(env, src, X, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, dst, Y, 0);
}


JNIEXPORT void JNICALL Java_jnn_core_JNNnative_sigmoidd(
    JNIEnv * env,
    jclass cls,
    jfloatArray sig,
    jfloatArray g,
    jfloatArray dst,
    jint tam
) {
    (void) cls;

    float* restrict SIG = (*env)->GetPrimitiveArrayCritical(env, sig, NULL);
    float* restrict G   = (*env)->GetPrimitiveArrayCritical(env, g, NULL);
    float* restrict DST = (*env)->GetPrimitiveArrayCritical(env, dst, NULL);

    jnn_sigmoid_d(SIG, G, DST, (size_t) tam);

    (*env)->ReleasePrimitiveArrayCritical(env, sig, (void*)SIG, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, g, (void*)G, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, dst, DST, 0);
}