package jnn.train.callback;

/**
 * Classe base para uso em callbacks de treino.
 */
public final class EpochInfo {

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
    public EpochInfo(int epoca, float perda) {
        this.epoca = epoca;
        this.perda = perda;
    }
}
