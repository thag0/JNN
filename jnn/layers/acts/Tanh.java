package jnn.layers.acts;

import jnn.core.tensor.Tensor;

/**
 * Camada que opera a função de ativação Tanh (Tangente Hiperbólica).
 */
public class Tanh extends Activation {
    
    /**
     * Inicializa uma camada de ativação Tanh.
     */
    public Tanh() {}
    
    /**
     * Inicializa uma camada de ativação Tanh.
     * @param input formato de entrada para a camada.
     */
    public Tanh(int... input) {
        build(input);
    }

    @Override
    public Tensor forward(Tensor x) {
        isBuilt();

		maybeFitToBatch(x);

		_input = x.contiguous();

        tanh(
            _input.array(),
            _output.array()
        );

        return _output;
    }

    @Override
    public Tensor backward(Tensor g) {
		isBuilt();

		_gradOutput = g.contiguous();

        tanhd(
            _output.array(), // tanh já calculada
            _gradOutput.array(),
            _gradInput.array()
        );

        return _gradInput;
    }

    private void tanh(float[] input, float[] output) {
        final int size = input.length;

        for (int i = 0; i < size; i++) {
            float val = input[i];
            output[i] = (float) (2.0 / (1.0 + Math.exp(-2.0 * val)) - 1.0);
        }
    }

    private void tanhd(float[] tanh, float[] gOut, float[] gradE) {
        final int size = tanh.length;

        for (int i = 0; i < size; i++) {
            final float g = gOut[i];
            final float t = tanh[i];
            final float d = 1.0f - (t * t);
            gradE[i] = g * d;
        }
    }
    
    @Override
    public Tanh clone() {
        return (Tanh) super.clone();
    }

}
