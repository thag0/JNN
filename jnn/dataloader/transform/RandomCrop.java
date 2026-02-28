package jnn.dataloader.transform;

import java.util.Random;
import jnn.core.tensor.Tensor;

/**
 * Transformação de corte aleatório (random crop) para imagens no formato CHW,
 * com opção de preenchimento (padding).
 */
public class RandomCrop implements Transform {

    private final int altS;
    private final int largS;
    private final int pad;
    private final Random rng = new Random();
    private final boolean reflect;

    /**
     * Inicializa uma transformação de corte aleatório.
     * @param altS Altura da imagem de saída.
     * @param largS Largura da imagem de saída.
     * @param pad Preenchimento (padding) aplicado antes do corte.
     * @param reflect Se verdadeiro, aplica reflexão ao preencher.
     */
    public RandomCrop(int altS, int largS, int pad, boolean reflect) {
        if (altS < 1 || largS < 1) {
            throw new IllegalArgumentException(
                "\nA altura e largura de saída devem ser maiores que zero."
            );
        }

        if (pad < 0) {
            throw new IllegalArgumentException(
                "\nO preenchimento (padding) deve ser maior ou igual a zero."
            );
        }

        this.altS = altS;
        this.largS = largS;
        this.pad = pad;
        this.reflect = reflect;
    }

    /**
     * Inicializa uma transformação de corte aleatório sem reflexão.
     * @param altS Altura da imagem de saída.
     * @param largS Largura da imagem de saída.
     * @param pad Preenchimento (padding) aplicado antes do corte.
     */
    public RandomCrop(int altS, int largS, int pad) {
        this(altS, largS, pad, false);
    }

    @Override
    public Tensor apply(Tensor t) {
        int[] s = t.shape();
        if (s.length != 3) {
            throw new IllegalArgumentException("\nEsperado tensor 3D (CHW)");
        }    

        final int canais = s[0];
        final int altura = s[1];
        final int largura = s[2];

        final int altP  = altura  + 2 * pad;
        final int largP = largura + 2 * pad;

        int y0 = rng.nextInt(altP - altS + 1);
        int x0 = rng.nextInt(largP - largS + 1);

        Tensor out = new Tensor(canais, altS, largS);

        float[] x = t.array();
        float[] y = out.array();

        final int areaX = altura * largura;
        final int areaS = altS * largS;

        if (!reflect) {
            for (int c = 0; c < canais; c++) {
                int baseX = c * areaX;
                int baseY = c * areaS;

                for (int h = 0; h < altS; h++) {
                    int srcY = y0 + h - pad;
                    if (srcY < 0 || srcY >= altura) continue;
                    int dstRow = baseY + h * largS;

                    for (int w = 0; w < largS; w++) {
                        int srcX = x0 + w - pad;
                        if (srcX < 0 || srcX >= largura) continue;

                        y[dstRow + w] = x[baseX + srcY * largura + srcX];
                    }
                }
            }

        } else {
            for (int c = 0; c < canais; c++) {
                int baseX = c * areaX;
                int baseY = c * areaS;

                for (int h = 0; h < altS; h++) {
                    int idY = reflect(y0 + h - pad, altura);
                    int linX = baseX + idY * largura;
                    int linY = baseY + h * largS;

                    for (int w = 0; w < largS; w++) {
                        int idX = reflect(x0 + w - pad, largura);
                        y[linY + w] = x[linX + idX];
                    }
                }
            }
        }

        return out;
    }

    /**
     * Reflete a coordenada x dentro do intervalo [0, size-1].
     * @param x Coordenada a ser refletida.
     * @param size Tamanho do intervalo.
     * @return Coordenada refletida.
     */
    private int reflect(int x, int size) {
        while (x < 0 || x >= size) {
            if (x < 0) x = -x - 1;
            else x = 2 * size - x - 1;
        }

        return x;
    }

}