package jnn.layers.acts;

import jnn.core.JNNnative;
import jnn.core.tensor.Tensor;

/**
 * Camada que opera a função de ativação Sigmoid.
 */
public class Sigmoid extends Activation {

    /**
     * Inicializa uma camada de ativação Sigmoid.
     */
    public Sigmoid() {}
    
    /**
     * Inicializa uma camada de ativação Sigmoid.
     * @param input formato de entrada para a camada.
     */
    public Sigmoid(int... input) {
        build(input);
    }

    @Override
    public Tensor forward(Tensor x) {
        validateBuid();

		maybeFitToBatch(x);

		_input = x.contiguous();

        if (JNNnative.isOn()) {
            JNNnative.sigmoid(
                _input.array(),
                _output.array(),
                _input.len()
            );
        } else {
            sigmoid(
                _input.array(),
                _output.array()
            );
        }

        return _output;
    }

    @Override
    public Tensor backward(Tensor g) {
		validateBuid();

		_gradOutput = g.contiguous();

        if (JNNnative.isOn()) {
            JNNnative.sigmoidd(
                _output.array(),
                _gradOutput.array(),
                _gradInput.array(),
                _input.len()
            );
        } else {
            sigmoidd(
                _output.array(),
                _gradOutput.array(),
                _gradInput.array()
            );
        }

        return _gradInput;
    }

    private void sigmoid(float[] input, float[] output) {
        final int size = input.length;

        for (int i = 0; i < size; i++) {
            float val = input[i];
            output[i] = (float) (1.0f / (1.0 + Math.exp(-val)));
        }
    }

    private void sigmoidd(float[] sig, float[] gOut, float[] gIn) {
        final int size = sig.length;

        for (int i = 0; i < size; i++) {
            final float g = gOut[i];
            final float s = sig[i];
            gIn[i] = g * (s * (1f - s));
        }
    }
    
    @Override
    public Sigmoid clone() {
        return (Sigmoid) super.clone();
    }

}
