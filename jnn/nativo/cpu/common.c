#include "common.h"

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
