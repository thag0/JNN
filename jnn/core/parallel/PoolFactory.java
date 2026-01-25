package jnn.core.parallel;

import java.util.concurrent.ForkJoinPool;

/**
 * Molde de criação para multiprocessamento.
 */
public class PoolFactory {
    
    /**
     * Máxima quantidade de threads disponível para a JVM.
     */
    static final int MAX_DISPONIVEL = Runtime.getRuntime().availableProcessors();

    /**
     * Quantidade de threads padrão por pool.
     */
    static int numThreads = MAX_DISPONIVEL / 2;// normalmente threads físicas.


    /**
     * Cria uma nova pool de threads utilizando o valor padrão.
     * @return {@code ForkJoinPool}
     */
    public static synchronized ForkJoinPool pool() {
        return new ForkJoinPool(numThreads);
    }

    /**
     * Cria uma nova pool de threads utilizando o valor padrão.
     * @param t número de threads desejadas para pool.
     * @return {@code ForkJoinPool}
     */
    public static synchronized ForkJoinPool pool(int t) {
        return new ForkJoinPool(t);
    }

    /**
     * Configura um novo valor padrão para criação de novas pools.
     * @param t quantidade de threads desejada.
     */
    public static synchronized void setThreads(int t) {
        if (t < 1) {
            throw new IllegalArgumentException(
                "\nValor de threads " + t + " inválido."
            );
        }

        numThreads = t;
    }

    /**
     * Retorna a quantidade de threads configurada para novas pools.
     * @return número de threads.
     */
    public static synchronized int getThreads() {
        return numThreads;
    }

}