package jnn.dataloader.transform;

import jnn.core.tensor.Tensor;

/**
 * Transformação de normalização.
 */
public class TNorm implements Transform {

    private Number min;
    private Number max;

    /**
     * Inicializa um transformador utilizando valores 
     * de mínimo e máximo desejados.
     * @param min valor mínimo.
     * @param max valor máximo.
     */
    public TNorm(Number min, Number max) {
        this.min = min;
        this.max = max;
    }

    /**
     * Inicializa um transformador utilizando valores 
     * de mínimo = 0 e máximo = 1.
     */
    public TNorm() {
        this(0.0, 1.0);
    }

    @Override
    public Tensor apply(Tensor t) {
        return t.norm(min, max);
    }
    
}
