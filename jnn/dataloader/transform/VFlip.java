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
    public void apply(Tensor in, Tensor out) {
        validarShapes(in, out);

        if (JNNutils.randFloat() > p) {
            out.copiar(in);
            return ;
        }

        int[] shape = in.shape();
        if (shape.length != 3) {
            throw new IllegalArgumentException("\nEsperado tensor 3D (CHW)");
        }

        final int canais  = shape[0];
        final int altura  = shape[1];
        final int largura = shape[2];

        float[] src = in.array();
        float[] dst = out.array();

        final int area = altura * largura;
        for (int c = 0; c < canais; c++) {
            int base = c * area;

            for (int h = 0; h < altura; h++) {
                int cima = base + h * largura;
                int baixo = base + (altura - 1 - h) * largura;

                for (int w = 0; w < largura; w++) {
                    dst[cima + w] = src[baixo + w];
                }
            }
        }

    }
}