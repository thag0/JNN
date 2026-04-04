package jnn.core.parallel;

import java.util.concurrent.ForkJoinPool;

import jnn.core.JNNlog;
import jnn.core.JNNlog.TipoLog;

// Ainda quero mudar como isso funciona
// Comparado ao OpenMP usado na parte nativa a diferença é absurda
// Tenho ideias de criar uma propria "ThreadPool" usando "Thread" manual e gerenciar por aqui

/**
 * Interface de paralelismo.
 */
public class JNNparallel {
    
    /**
     * Máxima quantidade de threads disponível para a JVM.
     */
    static final int MAX_DISPONIVEL = Runtime.getRuntime().availableProcessors();

    /**
     * Quantidade de threads padrão por pool.
     */
    static volatile int numThreads = MAX_DISPONIVEL / 2;// normalmente threads físicas.

    /**
     * Pool global.
     */
    static final ForkJoinPool common = pool(numThreads); 

    /**
     * Construtor privado.
     */
    private JNNparallel() {}

    /**
     * Cria uma nova pool de threads utilizando o valor padrão.
     * @param t número de threads desejadas para pool.
     * @return {@code ForkJoinPool}
     */
    public static ForkJoinPool pool(int t) {
        return new ForkJoinPool(t);
    }

    /**
     * Cria uma nova pool de threads utilizando o valor padrão.
     * @return {@code ForkJoinPool}
     */
    public static ForkJoinPool pool() {
        return pool(numThreads);
    }

    /**
     * Retorna a pool global.
     * @return pool global.
     */
    public static ForkJoinPool common() {
        return common;
    }

    /**
     * Configura um novo valor padrão para criação de novas pools.
     * @param t quantidade de threads desejada.
     */
    public static void setThreads(int t) {
        if (t < 1) {
            throw new IllegalArgumentException(
                "\nValor de threads " + t + " inválido."
            );
        }

        if (t > MAX_DISPONIVEL) {
            JNNlog.logln(
                TipoLog.PARALLEL,
                "Threads configuradas " + t + " é maior que o valor disponível " + MAX_DISPONIVEL
                + ", podendo afetar o desempenho."
            );
        }

        numThreads = t;
    }

    /**
     * Retorna a quantidade de threads configurada para novas pools.
     * @return número de threads.
     */
    public static int getThreads() {
        return numThreads;
    }

}