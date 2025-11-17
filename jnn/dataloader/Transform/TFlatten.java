package jnn.dataloader.transform;

import jnn.core.tensor.Tensor;

/**
 * Transformação de achatamento.
 */
public class TFlatten implements Transform {

    @Override
    public Tensor apply(Tensor t) {
        return t.flatten();
    }
    
}
