package jnn.train.callback;

/**
 * Interface para uso de callbacks durante os treinos de modelos.
 */
@FunctionalInterface
public interface Callback {
    
    /**
     * Executa o callback utilizando os dados da época atual.
     * @param info informações sobre a época.
     */
    public void onEpochEnd(EpochInfo info);

}
