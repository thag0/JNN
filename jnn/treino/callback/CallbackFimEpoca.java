package jnn.treino.callback;

/**
 * Interface para uso de callbacks durante os treinos de modelos.
 */
@FunctionalInterface
public interface CallbackFimEpoca {
    
    /**
     * Executa o callback utilizando os dados da época atual.
     * @param info informações sobre a época.
     */
    public void run(InfoEpoca info);

}
