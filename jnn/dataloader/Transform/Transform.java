package jnn.dataloader.transform;

import jnn.core.tensor.Tensor;

/**
 * Inferface para aplicar transformações em {@code Tensor} dentro
 * do {@code DataLoader}
 * @see {@code Tensor} {@link jnn.core.tensor.Tensor}
 * @see {@code DataLoader} {@link jnn.dataloader.DataLoader}
 */
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

}
