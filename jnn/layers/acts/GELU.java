package jnn.layers.acts;

import jnn.core.tensor.Tensor;

/**
 * Camada que opera a função de ativação GELU.
 */
public class GELU extends Activation {

    /**
     * Constante pré calculada.
     */
	private final float RAIZ_2_POR_PI = (float) Math.sqrt(2 / Math.PI); 

    /**
     * Constante empírica usada na aproximação da função junto da tanh.
     */
	private final float C = 0.044715f;
    
    /**
     * Inicializa uma camada de ativação GELU.
     * @param input formato de entrada para a camada.
     */
    public GELU(int... input) {
        build(input);
    }
 
    /**
     * Inicializa uma camada de ativação GELU.
     */
    public GELU () {}

    @Override
    public Tensor forward(Tensor x) {
        validateBuid();

		maybeFitToBatch(x);

		_input = x.contiguous();

        gelu(
            _input.array(),
            _output.array()
        );

        return _output;
    }

    @Override
    public Tensor backward(Tensor g) {
		validateBuid();

		_gradOutput = g.contiguous();

        gelud(
            _input.array(),
            _gradOutput.array(),
            _gradInput.array()
        );

        return _gradInput;
    }

    private void gelu(float[] input, float[] output) {
        final int size = input.length;

        for (int i = 0; i < size; i++) {
            float x = input[i];
            float u = RAIZ_2_POR_PI * (x + C * (x*x*x));
            float t = tanh(u);

            output[i] = 0.5f * x * (1.0f + t);
        }
    }

    private void gelud(float[] input, float[] gOut, float[] gIn) {
        final int size = input.length;

        for (int i = 0; i < size; i++) {
            float x = input[i];
            float v2 = x * x;
            float v3 = v2 * x;
            float u = RAIZ_2_POR_PI * (x + C * v3);
            float t = tanh(u);
            float dt = 1.0f - t * t;
            float du = RAIZ_2_POR_PI * (1.0f + 3.0f * C * v2);
            float dgelu = 0.5f * (1.0f + t) + 0.5f * x * dt * du;

            gIn[i] = gOut[i] * dgelu;
        }
    }

    private float tanh(float x) {
        return (float) (2.0 / (1.0 + Math.exp(-2.0 * x)) - 1.0);
    }

}
