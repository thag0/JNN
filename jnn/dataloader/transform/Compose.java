package jnn.dataloader.transform;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

/**
 * Compositor
 * <p>
 *      Age aplicando uma série de transformações em sequência.
 * </p>
 */
public class Compose implements Transform {

    /**
     * Sequência de transformações.
     */
    private final Transform[] ts;

    /**
     * Inicializa um novo compositor de transformações.
     * @param ts transformações para aplicação.
     */
    public Compose(Transform... ts) {
        JNNutils.validarNaoNulo(ts, "transforms == null");

        if (ts.length == 0) {
            throw new IllegalArgumentException(
                "\nNenhuma transformação fornecida para o compositor."
            );
        }
        
        for (var t : ts) {
            JNNutils.validarNaoNulo(t, "transform == null");
        }

        this.ts = ts;
    }

    @Override
    public Tensor apply(Tensor t) {
        Tensor out = t.clone();

        for (Transform tr : ts) {
            out = tr.apply(out);
        }

        return out;
    }
    
}
