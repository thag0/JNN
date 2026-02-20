package jnn.treino.callback;

/**
 * Classe base para uso em callbacks de treino.
 */
public final class InfoEpoca {

    /**
     * Índice da época atual.
     */
    public final int epoca;

    /**
     * Valor da perda atual.
     */
    public final float perda;

    /**
     * Inicializa uma nova informação sobre a época de treino.
     * @param epoca índice da época atual.
     * @param perda valor de perda da época atual.
     */
    public InfoEpoca(int epoca, float perda) {
        this.epoca = epoca;
        this.perda = perda;
    }
}
