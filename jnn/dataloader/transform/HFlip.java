package jnn.dataloader.transform;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

/**
 * Transformação de flip horizontal (espelhamento) para imagens no formato CHW,
 * aplicada com uma certa probabilidade.
 */
public class HFlip implements Transform {

    /**
     * Chance de aplicar o flip horizontal.
     */
    private final float p;

    /**
     * Inicializa uma transformação de flip horizontal.
     * @param p Probabilidade de aplicação.
     */
    public HFlip(double p) {
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
            return;
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
            int baseC = c * area;

            for (int h = 0; h < altura; h++) {
                int linha = baseC + h * largura;
                int esq = linha;
                int dir = linha + largura - 1;

                while (esq < dir) {
                    dst[esq] = src[dir];
                    dst[dir] = src[esq];

                    esq++;
                    dir--;
                }
            }
        }

    }

}
