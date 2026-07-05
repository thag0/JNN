package jnn.dataloader.transform;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

/**
 * Transformação de variação de cor aplicadas com uma certa probabilidade.
 */
public class ColorJitter implements Transform {

    private final float PROB;
    private final float BRILHO;
    private final float CONTRASTE;
    private final float SATURACAO;

    /**
     * Inicializa uma transformação de variação de cor.
     * @param brilho taxa de variação de brilho.
     * @param contraste taxa de variação no constraste.
     * @param saturacao taxa de variação de saturação.
     * @param prob taxa de atualização dos parâmetros.
     */
    public ColorJitter(Number brilho, Number contraste, Number saturacao, Number prob) {
        this.BRILHO = brilho.floatValue();
        this.CONTRASTE = contraste.floatValue();
        this.SATURACAO = saturacao.floatValue();
        this.PROB = prob.floatValue();

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
        
        if (PROB < 0 || PROB > 1) {
            throw new IllegalArgumentException(
                "\nTaxa de probabilidade deve estar entre 0 e 1."
            );
        }
    }

    @Override
    public Tensor apply(Tensor t) {
        if (JNNutils.randFloat() > PROB) return t;

        Tensor out = t.clone();

        if (BRILHO > 0) ajustarBrilho(out);
        if (CONTRASTE > 0) ajustarContraste(out);
        if (SATURACAO > 0) ajustarSaturacao(out);

        return out;
    }

    /**
     * Aplica as correções que afetam o brilho.
     * @param t {@code Tensor} base.
     */
    private void ajustarBrilho(Tensor t) {
        float fator = 1.f + (JNNutils.randFloat() * 2 - 1) * BRILHO;
        float[] x = t.array();

        for (int i = 0; i < x.length; i++) {
            x[i] *= fator;

            if (x[i] < 0) x[i] = 0;
            if (x[i] > 1) x[i] = 1;
        }
    }

    /**
     * Aplica as correções que afetam o contraste.
     * @param t {@code Tensor} base.
     */
    private void ajustarContraste(Tensor t) {
        float fator = 1.f + (JNNutils.randFloat() * 2 - 1) * CONTRASTE;
        float[] x = t.array();

        float media = 0f;
        for (float val : x) media += val;
        media /= x.length;

        for (int i = 0; i < x.length; i++) {
            x[i] = (x[i] - media) * fator + media;

            if (x[i] < 0) x[i] = 0;
            if (x[i] > 1) x[i] = 1;           
        }
    }

    /**
     * Aplica as correções que afetam a saturação.
     * @param t {@code Tensor} base.
     */
    private void ajustarSaturacao(Tensor t) {
        int[] shape = t.shape();

        if (shape[0] != 3) throw new RuntimeException("\nEro");

        float fator = 1f + (JNNutils.randFloat() * 2 - 1) * SATURACAO;
    
        int h = shape[1];
        int w = shape[2];
        int area = h * w;

        float[] x = t.array();

        for (int i = 0; i < area; i++) {
            float r = x[i];
            float g = x[i + area];
            float b = x[i + 2 * area];
        
            float cinza = 0.299f * r + 0.587f * g + 0.114f * b;

            x[i]          = cinza + fator * (r - cinza);
            x[i + area]   = cinza + fator * (g - cinza);
            x[i + 2*area] = cinza + fator * (b - cinza);

            x[i]          = Math.max(0, Math.min(1, x[i]));
            x[i + area]   = Math.max(0, Math.min(1, x[i + area]));
            x[i + 2*area] = Math.max(0, Math.min(1, x[i + 2*area]));
        }
    }
    
}
