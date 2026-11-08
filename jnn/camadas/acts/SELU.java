package jnn.camadas.acts;

import jnn.core.tensor.Tensor;

/**
 * Camada que opera a função de ativação SELU.
 */
public class SELU extends Activation {

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
     * @param input formato de entrada para a camada.
     * @param alpha parâmetro para controlar curvas negativas.
     * @param gamma parâmetro para controlar a escala de saída.
     */
    public SELU(int[] input, Number alpha, Number gamma) {
        this(alpha, gamma);
        build(input);
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
     * @param input formato de entrada para a camada.
     */
    public SELU(int... input) {
        build(input);
    }

    /**
     * Inicializa uma camada de ativação SELU.
     */
    public SELU() {}

    @Override
    public Tensor forward(Tensor x) {
        isBuilt();

		maybeFitToBatch(x);

		_input = x.contiguous();

        selu(
            _input.array(),
            _output.array()
        );

        return _output;
    }

    @Override
    public Tensor backward(Tensor g) {
		isBuilt();

		_gradOutput = g.contiguous();

        selud(
            _input.array(),
            _gradOutput.array(), 
            _gradInput.array()
        );

        return _gradInput;
    }

    private void selu(float[] input, float[] output) {
        final int size = input.length;

        for (int i = 0; i < size; i++) {
            float val = input[i];
            output[i] = (val > 0.0) ? gamma * val : gamma * alpha * (float) (Math.exp(val) - 1.0);
        }
    }

    private void selud(float[] input, float[] gOut, float[] gIn) {
        final int size = input.length;

        for (int i = 0; i < size; i++) {
            final float g = gOut[i];
            final float x = input[i];
            gIn[i] = g * ((x > 0.0) ? gamma : gamma * alpha * (float) Math.exp(x));
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
