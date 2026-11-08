package jnn.layers.acts;

import jnn.core.tensor.Tensor;

/**
 * Camada que opera a função de ativação Swish.
 */
public class Swish extends Activation {

    /**
     * Inicializa uma camada de ativação Swish.
     */
    public Swish() {}

    /**
     * Inicializa uma camada de ativação ReLU.
     * @param input formato de entrada para a camada.
     */
    public Swish(int... input) {
        build(input);
    }

    @Override
    public Tensor forward(Tensor x) {
        validateBuid();

		maybeFitToBatch(x);

		_input = x.contiguous();

        relu(
            _input.array(), 
            _output.array()
        );

        return _output;
    }

    @Override
    public Tensor backward(Tensor g) {
		validateBuid();

		_gradOutput = g.contiguous();

        relud(
            _input.array(),
            _gradOutput.array(),
            _gradInput.array()
        );

        return _gradInput;
    }

    private void relu(float[] input, float[] output) {
        final int size = input.length;

        for (int i = 0; i < size; i++) {
            float val = input[i];
            output[i] = val * sigmoid(val);
        }
    }

    private void relud(float[] input, float[] gOut, float[] gIn) {
        final int n = input.length;

        for (int i = 0; i < n; i++) {
            final float g = gOut[i];
            final float x = input[i];
            final float sig = sigmoid(x);
            gIn[i] = g * (sig + (x * sig * (1.0f - sig)));
        }
    }

	final private float sigmoid(float x) {
		return (float) (1.0 / (1.0 + Math.exp(-x)));
	}
    
    @Override
    public Swish clone() {
        return (Swish) super.clone();
    }

}
