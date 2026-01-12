#ifndef JNN_NATIVE_COMMON_H
    #define JNN_NATIVE_COMMON_H

    #include <jni.h>
    #include <omp.h>

    static inline int jnn_native_num_threads() {
        int p = omp_get_num_procs();
        return p > 1 ? p / 2 : 1;
    }

    JNIEXPORT void JNICALL
    Java_jnn_nativo_JNNNative_setThreads(
        JNIEnv* env, jclass cls, jint n
    );

#endif
