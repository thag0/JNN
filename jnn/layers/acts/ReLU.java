package jnn.layers.acts;

import jnn.core.JNNnative;
import jnn.core.tensor.Tensor;

/**
 * Camada de ativação ReLU.
 */
public class ReLU extends Activation {

    /**
     * Inicializa uma camada de ativação ReLU.
     */
    public ReLU() {}

    /**
     * Inicializa uma camada de ativação ReLU.
     * @param input formato de entrada para a camada.
     */
    public ReLU(int... input) {
        build(input);
    }

    @Override
    public Tensor forward(Tensor x) {
        validateBuid();

		maybeFitToBatch(x);

		_input = x.contiguous();

        if (JNNnative.isOn()) {
            JNNnative.relu(
                _input.array(),
                _output.array(),
                _input.len()
            );
        } else {
            relu(
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
            JNNnative.relud(
                _input.array(), 
                _gradOutput.array(), 
                _gradInput.array(), 
                _input.len()
            );
        } else {
            relud(
                _input.array(),
                _gradOutput.array(), 
                _gradInput.array()
            );
        }

        return _gradInput;
    }

    private static void relu(float[] input, float[] output) {
        final int size = input.length;

        for (int i = 0; i < size; i++) {
            float val = input[i];
            output[i] = val > 0 ? val : 0;
        }
    }

    private static void relud(float[] input, float[] gOut, float[] gIn) {
        final int size = input.length;

        for (int i = 0; i < size; i++) {
            final float g = gOut[i];
            final float x = input[i];
            gIn[i] = g * (x > 0 ? 1 : 0);
        }
    }
    
    @Override
    public ReLU clone() {
        return (ReLU) super.clone();
    }

}
