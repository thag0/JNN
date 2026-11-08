package jnn.layers.acts;

import jnn.core.tensor.Tensor;

/**
 * Camada de ativação Leaky ReLU.
 */
public class LeakyReLU extends Activation {

    /**
     * Parâmetro personalizável.
     */
    private float alpha;

    /**
     * Inicializa uma camada de ativação Leaky ReLU.
     * @param alpha constante da ativação.
     * @param input formato de entrada para a camada.
     */
    public LeakyReLU(Number alpha, int... input) {
        this(alpha);
        build(input);
    }

    /**
     * Inicializa uma camada de ativação Leaky ReLU.
     * @param input formato de entrada para a camada.
     */
    public LeakyReLU(int... input) {
        build(input);
    }

    /**
     * Inicializa uma camada de ativação Leaky ReLU.
     * @param alpha constante da ativação.
     */
    public LeakyReLU(Number alpha) {
        this.alpha = alpha.floatValue();
    }

    /**
     * Inicializa uma camada de ativação Leaky ReLU.
     */
    public LeakyReLU() {
        this(0.01);
    }

    @Override
    public Tensor forward(Tensor x) {
        validateBuid();

        maybeFitToBatch(x);

		_input = x.contiguous();

        leakyrelu(
            _input.array(),
            _output.array()
        );

        return _output;
    }

    @Override
    public Tensor backward(Tensor g) {
		validateBuid();

		_gradOutput = g.contiguous();

        leakyrelud(
            _input.array(),
            _gradOutput.array(),
            _gradInput.array() 
        );

        return _gradInput;
    }

    private void leakyrelu(float[] input, float[] output) {
        final int size = input.length;

        for (int i = 0; i < size; i++) {
            float val = input[i];
            output[i] = val > 0.0 ? val : val * alpha;
        }
    }

    private void leakyrelud(float[] input, float[] gOut, float[] gIn) {
        final int size = input.length;

        for (int i = 0; i < size; i++) {
            final float g = gOut[i];
            final float x = input[i];
            gIn[i] = g * (x > 0f ? 1f : alpha);
        }
    }
    
    @Override
    public LeakyReLU clone() {
        return (LeakyReLU) super.clone();
    }
    
    /**
     * Retorna o parâmetro da camada.
     * @return constante alpha.
     */
    public float getAlpha() {
        return alpha;
    }

}
