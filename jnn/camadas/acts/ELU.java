package jnn.camadas.acts;

import jnn.core.tensor.Tensor;

/**
 * Camada que opera a função de ativação ELU.
 */
public class ELU extends Activation {
    
	/**
	 * Parâmetro da camada.
	 */
	private float alpha = 0.01f;

	/**
	 * Instancia a função de ativação ELU com 
	 * seu valor de alfa configurável.
	 * @param alpha novo valor alpha.
     * @param input formato de entrada da camada.
	 */
	public ELU(Number alpha, int... input) {
		this(alpha);
        build(input);
	}

	/**
	 * Instancia a função de ativação ELU com 
	 * seu valor de alfa configurável.
	 * @param alpha novo valor alpha.
	 */
	public ELU(Number alpha) {
		this.alpha = alpha.floatValue();
	}

	/**
	 * Instancia a função de ativação ELU com 
	 * seu valor de alpha padrão.
	 * <p>
	 *    O valor padrão para o alpha é {@code 0.01}.
	 * </p>
	 */
	public ELU() {
		this(0.01);
	}

    @Override
    public Tensor forward(Tensor x) {
        isBuilt();

        maybeFitToBatch(x);

		_input = x.contiguous();

        elu(
            _input.array(),
            _output.array()
        );

        return _output;
    }

    @Override
    public Tensor backward(Tensor g) {
		isBuilt();

		_gradOutput = g.contiguous();

        elud(
            _input.array(),
            _gradOutput.array(),
            _gradInput.array()
        );

        return _gradInput;
    }

    private void elu(float[] input, float[] output) {
        final int size = input.length;

        for (int i = 0; i < size; i++) {
            float val = input[i];
            output[i] = (val > 0) ? val : alpha * (float) (Math.exp(val) - 1.0);
        }
    }

    private void elud(float[] input, float[] gOut, float[] gIn) {
        final int size = input.length;

        for (int i = 0; i < size; i++) {
            final float g = gOut[i];
            final float x = input[i];
            gIn[i] = g * ((x > 0.0) ? 1.0f : alpha * (float) Math.exp(x));
        }
    }

    /**
     * Retorna o valor do parâmetro da camada.
     * @return constante alfa.
     */
    public float getAlpha() {
        return alpha;
    }

}
