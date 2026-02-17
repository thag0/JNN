package jnn.dataloader.transform;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

/**
 * Compositor
 * <p>
 *      Age aplicando uma série de transformações em sequência.
 * </p>
 */
public class TCompose implements Transform {

    /**
     * Sequência de transformações.
     */
    private final Transform[] transforms;

    /**
     * Inicializa um novo compositor de transformações.
     * @param ts transformações para aplicação.
     */
    public TCompose(Transform... ts) {
        JNNutils.validarNaoNulo(ts, "transforms == null");
        
        for (var t : ts) {
            if (t == null) {
                throw new IllegalArgumentException(
                    "\nCompose não pode receber transformações nulas."
                );
            }
        }

        this.transforms = ts;
    }

    @Override
    public Tensor apply(Tensor t) {
        Tensor out = t.clone();

        for (Transform tr : transforms) {
            out = tr.apply(out);
        }

        return out;
    }
    
}
