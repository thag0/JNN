package jnn.dataloader.transform;

import jnn.core.tensor.Tensor;

/**
 * Inferface para aplicar transformações em {@code Tensor} dentro
 * do {@code DataLoader}
 * @see jnn.core.tensor.Tensor Tensor
 * @see jnn.dataloader.DataLoader DataLoader
 */
@FunctionalInterface
public interface Transform {
    
    /**
     * Aplica uma transformação no {@code Tensor}.
     * @param t {@code Tensor} base.
     * @return {@code Tensor} transformado.
     */
    Tensor apply(Tensor t);

    /**
     * Reaplica uma transformação no {@code Tensor}.
     * @param tr {@code Transform} que será aplicada na sequência.
     * @return {@code Tensor} transformado.
     */
    default Transform andThen(Transform tr) {
        return t -> tr.apply(apply(t));
    }

    /**
     * Retorna o nome da transformação.
     * @return Nome da transformação.
     */
    default public String nome() {
        return getClass().getSimpleName();
    }

}
