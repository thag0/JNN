package jnn.core;

public class JNNlog {

    /**
     * Tipos de logs disponíveis.
     */
    public enum TipoLog {
        TREINO,
        PARALLEL
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
    public static void logln(TipoLog level, String info) {
        log(level, info);
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
