package jnn.dataloader.transform;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

/**
 * Transformação de variação de cor aplicadas.
 */
public class ColorJitter implements Transform {

    private final float BRILHO;
    private final float CONTRASTE;
    private final float SATURACAO;

    /**
     * Inicializa uma transformação de variação de cor.
     * @param brilho taxa de variação de brilho.
     * @param contraste taxa de variação no constraste.
     * @param saturacao taxa de variação de saturação.
     */
    public ColorJitter(Number brilho, Number contraste, Number saturacao) {
        this.BRILHO = brilho.floatValue();
        this.CONTRASTE = contraste.floatValue();
        this.SATURACAO = saturacao.floatValue();

        if (BRILHO < 0) {
            throw new IllegalArgumentException(
                "\nValor de brilho deve ser maior que zero."
            );
        }
        
        if (CONTRASTE < 0) {
            throw new IllegalArgumentException(
                "\nValor de contraste deve ser maior que zero."
            );
        }
        
        if (SATURACAO < 0) {
            throw new IllegalArgumentException(
                "\nValor de saturação deve ser maior que zero."
            );
        }

    }

    @Override
    public Tensor apply(Tensor t) {
        int[] shape = t.shape();

        if (shape.length != 3 || shape[0] != 3) {
            throw new IllegalArgumentException(
                "ColorJitter espera um tensor RGB (3, H, W)."
            );
        }

        Tensor out = t.clone();

        float brilho = fator(BRILHO);
        float contraste = fator(CONTRASTE);
        float saturacao = fator(SATURACAO);

        int[] ordem = {0, 1, 2};
        for (int i = ordem.length - 1; i > 0; i--) {
            int j = (int) (JNNutils.randFloat() * (i + 1));
            if (j > i) j = i;

            int tmp = ordem[i];
            ordem[i] = ordem[j];
            ordem[j] = tmp;
        }

        for (int op : ordem) {
            switch (op) {
                case 0:
                    if (BRILHO > 0) ajustarBrilho(out, brilho);
                break;

                case 1:
                    if (CONTRASTE > 0) ajustarContraste(out, contraste);
                break;

                case 2:
                    if (SATURACAO > 0) ajustarSaturacao(out, saturacao);
                break;
            }
        }

        return out;
    }

    private float fator(float intensidade) {
        if (intensidade == 0) return 1f;

        return 1.0f + (2.0f * JNNutils.randFloat() - 1.0f) * intensidade;
    }

    /**
     * Trava o valor de entrada entre 0 e 1.
     * @param x entrada.
     * @return valor dentro do intervalo.
     */
    private float clamp(float x) {
        return Math.clamp(x, 0.0f, 1.0f);
    }

    /**
     * Aplica as correções que afetam o brilho.
     * @param t {@code Tensor} base.
     */
    private void ajustarBrilho(Tensor t, float fator) {
        float[] x = t.array();

        for (int i = 0; i < x.length; i++) {
            x[i] = clamp(x[i] * fator);
        }
    }

    /**
     * Aplica as correções que afetam o contraste.
     * @param t {@code Tensor} base.
     */
    private void ajustarContraste(Tensor t, float fator) {
        float[] x = t.array();

        float media = 0.0f;
        for (int i = 0; i < x.length; i++) {
            media += x[i];
        }
        media /= x.length;

        for (int i = 0; i < x.length; i++) {
            x[i] = clamp((x[i] - media) * fator + media);
        }
    }

    /**
     * Aplica as correções que afetam a saturação.
     * @param t {@code Tensor} base.
     */
    private void ajustarSaturacao(Tensor t, float fator) {
        int[] shape = t.shape();
        int area = shape[1] * shape[2];
        float[] x = t.array();

        for (int i = 0; i < area; i++) {
            float r = x[i];
            float g = x[i + area];
            float b = x[i + 2 * area];

            float cinza = 0.299f * r + 0.587f * g + 0.114f * b;

            x[i]          = clamp(cinza + fator * (r - cinza));
            x[i + area]   = clamp(cinza + fator * (g - cinza));
            x[i + 2*area] = clamp(cinza + fator * (b - cinza));
        }
    }
    
}
