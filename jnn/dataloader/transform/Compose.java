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
    public void apply(Tensor in, Tensor out) {
        validarShapes(in, out);

        int n = ts.length;

        if (n == 1) {
            ts[0].apply(in, out);
            return;
        }

        Tensor a = new Tensor(in);
        Tensor b = new Tensor(in);

        Tensor src = in;
        Tensor dst;

        for (int i = 0; i < n; i++) {
            if (i == n - 1) {
                dst = out;
            } else {
                dst = (i % 2 == 0) ? a : b;
            }

            ts[i].apply(src, dst);
            src = dst;
        }

    }
    
}