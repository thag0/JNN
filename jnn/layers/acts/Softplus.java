package jnn.layers.acts;

import jnn.core.tensor.Tensor;

/**
 * Camada que opera a função de ativação Softplus.
 */
public class Softplus extends Activation {
    
    /**
     * Inicializa uma camada de ativação Softplus.
     */
    public Softplus() {}

    /**
     * Inicializa uma camada de ativação Softplus.
     * @param input formato de entrada da camada.
     */
    public Softplus(int... input) {
        build(input);
    }

    @Override
    public Tensor forward(Tensor x) {
        isBuilt();

		maybeFitToBatch(x);

		_input = x.contiguous();

        softplus(
            _input.array(),
            _output.array()
        );

        return _output;
    }

    @Override
    public Tensor backward(Tensor g) {
		isBuilt();

		_gradOutput = g.contiguous();

        softplusd(
            _input.array(),
            _gradOutput.array(),
            _gradInput.array()
        );

        return _gradInput;
    }

    private void softplus(float[] input, float[] output) {
        final int size = input.length;

        for (int i = 0; i < size; i++) {
            float val = input[i];
            output[i] = (float) Math.log(1.0 + Math.exp(val));
        }
    }

    private void softplusd(float[] input, float[] gOut, float[] gradIn) {
        final int size = input.length;

        for (int i = 0; i < size; i++) {
            final float g = gOut[i];
            final float x = input[i];
            float exp = (float) Math.exp(x);
            gradIn[i] = g * (exp / (1.0f + exp));
        }
    }
    
    @Override
    public Softplus clone() {
        return (Softplus) super.clone();
    }

}
