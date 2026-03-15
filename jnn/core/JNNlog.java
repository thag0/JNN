package jnn.core;

/**
 * Serviço simples de loggin da biblioteca.
 */
public class JNNlog {

    /**
     * Tipos de logs disponíveis.
     */
    public enum TipoLog {

        /**
         * Logs relacionados ao treino.
         */
        TREINO,
        
        /**
         * Logs relacionados à paralelismo.
         */
        PARALLEL,

        /**
         * Logs relacionados à interface nativa.
         */
        NATIVO,

        /**
         * Logs relacionados aos datasets.
         */
        DATASET
    };
    
    /**
     * Construtor privado.
     */
    private JNNlog() {}

    /**
     * Exibe no console o log informado.
     * @param tipo tipo de informação.
     * @param info infomação para log.
     */
    public static void log(TipoLog tipo, String info) {
        if (info != null && tipo != null) {
            System.out.print("[" + tipo.toString() + "]: " + info);
        }
    }
    
    /**
     * Exibe no console o log informado, quebrando uma linha ao final.
     * @param tipo tipo de informação.
     * @param info infomação para log.
     */
    public static void logln(TipoLog tipo, String info) {
        log(tipo, info);
        System.out.printf("\n");
    }

    /**
     * Exibe um log específico para o treino.
     * @param info informação.
     */
    public static void logTreino(String info) {
        if (info != null) {
            System.out.print("\r" + info);
        }        
    }

}
