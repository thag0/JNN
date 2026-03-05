package jnn.dataloader.transform;

import jnn.core.tensor.Tensor;

/**
 * Transformação de suavização de rótulos para evitar valores "rígidos"
 * de 0 e 1, transformando em uma distibuição mais suave.
 */
public class LabelSmoothing implements Transform {

    /**
     * Fator de suavização.
     */
    private final float eps;

    /**
     * Inicializa uma transformação de suavização de rótulos.
     * @param eps Valor no intervalo [0, 1).
     */
    public LabelSmoothing(double eps) {
        if (eps < 0 || eps >= 1) {
            throw new IllegalArgumentException(
                "\nO epsilon deve estar no intervalo [0, 1), recebido " + eps
            );
        }

        this.eps = (float) eps;
    }

    @Override
    public Tensor apply(Tensor t) {
        int[] s = t.shape();
        if (s.length != 1) {
            throw new IllegalArgumentException(
                "\nEsperado tensor 1D representando probabilidades de classe."
            );
        }

        final int classes = s[0];

        Tensor out = new Tensor(t);
        float[] y = out.array();

        int alvo = 0;
        float max = y[0];
        for (int i = 1; i < classes; i++) {
            if (y[i] > max) {
                max = y[i];
                alvo = i;
            }
        }

        float valMenor = eps / (classes - 1);
        float valMaior  = 1f - eps;

        for (int i = 0; i < classes; i++) {
            y[i] = valMenor;
        }

        y[alvo] = valMaior;

        return out;
    }
}