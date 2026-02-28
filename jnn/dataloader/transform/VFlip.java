package jnn.dataloader.transform;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

/**
 * Transformação de flip vertical para imagens no formato CHW,
 * aplicada com uma certa probabilidade.
 */
public class VFlip implements Transform {

    /**
     * Chance de aplicar o flip vertical.
     */
    private final float p;
    
    /**
     * Inicializa uma transformação de flip vertical.
     * @param p Probabilidade de aplicação.
     */
    public VFlip(double p) {
        if (p < 0 || p > 1) {
            throw new IllegalArgumentException(
                "\nA probabilidade deve estar entre 0 e 1, recebido " + p
            );
        }

        this.p = (float) p;
    }

    @Override
    public Tensor apply(Tensor t) {
        if (JNNutils.randFloat() > p) return t;

        int[] s = t.shape();
        if (s.length != 3) {
            throw new IllegalArgumentException("\nEsperado tensor 3D (CHW)");
        }

        final int canais = s[0];
        final int altura = s[1];
        final int largura = s[2];

        Tensor out = new Tensor(t);
        float[] x = out.array();

        final int area = altura * largura;
        for (int c = 0; c < canais; c++) {
            int base = c * area;

            for (int h = 0; h < altura / 2; h++) {
                int cima = base + h * largura;
                int baixo = base + (altura - 1 - h) * largura;

                for (int w = 0; w < largura; w++) {
                    float tmp = x[cima + w];
                    x[cima + w] = x[baixo + w];
                    x[baixo + w] = tmp;
                }
            }
        }

        return out;
    }
}