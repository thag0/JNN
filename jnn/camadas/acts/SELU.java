package jnn.camadas.acts;

import jnn.core.tensor.Tensor;

/**
 * Camada que opera a função de ativação SELU.
 */
public class SELU extends Ativacao {

    /**
     * Parâmetro para controlar curvas negativas.
     */
    private float alpha = 1.6733f;
    
    /**
     * Parâmetro para controlar a escala de saída.
     */
    private float gamma = 1.0507f;
    
    /**
     * Inicializa uma camada de ativação SELU.
     * @param shape formato de entrada para a camada.
     * @param alpha parâmetro para controlar curvas negativas.
     * @param gamma parâmetro para controlar a escala de saída.
     */
    public SELU(int[] shape, Number alpha, Number gamma) {
        this(alpha, gamma);
        construir(shape);
    }
    
    /**
     * Inicializa uma camada de ativação SELU.
     * @param alpha parâmetro para controlar curvas negativas.
     * @param gamma parâmetro para controlar a escala de saída.
     */
    public SELU(Number alpha, Number gamma) {
        this.alpha = alpha.floatValue();
        this.gamma = gamma.floatValue();
    }

    /**
     * Inicializa uma camada de ativação SELU.
     * @param shape formato de entrada para a camada.
     */
    public SELU(int... shape) {
        construir(shape);
    }

    /**
     * Inicializa uma camada de ativação SELU.
     */
    public SELU() {}

    @Override
    public Tensor forward(Tensor x) {
        verificarConstrucao();

		talvesAjustarLote(x);

		_entrada = x.contiguous();

        selu(
            _entrada.array(), _entrada.offset(),
            _saida.array(), _saida.offset(),
            _entrada.tam()
        );

        return _saida;
    }

    @Override
    public Tensor backward(Tensor g) {
		verificarConstrucao();

		_gradSaida = g.contiguous();

        selud(
            _gradSaida.array(), _gradSaida.offset(),
            _gradEntrada.array(), _gradEntrada.offset(),
            _entrada.array(), _entrada.offset(),
            _entrada.tam()
        );

        return _gradEntrada;
    }

    private void selu(float[] x, int offX, float[] dest, int offDest, int n) {
        for (int i = 0; i < n; i++) {
            float val = x[offX + i];
            dest[offDest + i] = (val > 0.0) ? gamma * val : gamma * alpha * (float) (Math.exp(val) - 1.0);
        }
    }

    private void selud(float[] g, int offG, float[] gradE, int offGE, float[] x, int offX, int n) {
        for (int i = 0; i < n; i++) {
            final float grad = g[offG + i];
            final float entrada = x[offX + i];
            gradE[offGE + i] = grad * ((entrada > 0.0) ? gamma : gamma * alpha * (float) Math.exp(entrada));
        }
    }
    
    @Override
    public SELU clone() {
        return (SELU) super.clone();
    }

    /**
     * Retorna o parâmetro {@code alpha}.
     * @return {@code alpha}.
     */
    public float getAlpha() {
        return alpha;
    }
    
    /**
     * Retorna o parâmetro {@code gamma}.
     * @return {@code gamma}.
     */
    public float getGamma() {
        return gamma;
    }

}
