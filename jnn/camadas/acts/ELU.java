package jnn.camadas.acts;

import jnn.core.tensor.Tensor;

/**
 * Camada que opera a função de ativação ELU.
 */
public class ELU extends Ativacao {
    
	/**
	 * Parâmetro da camada.
	 */
	private float alpha = 0.01f;

	/**
	 * Instancia a função de ativação ELU com 
	 * seu valor de alfa configurável.
	 * @param alpha novo valor alpha.
     * @param shape formato de entrada da camada.
	 */
	public ELU(Number alpha, int... shape) {
		this(alpha);
        construir(shape);
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
        verificarConstrucao();

        talvesAjustarLote(x);

		_entrada = x.contiguous();

        elu(
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

        elud(
            _gradSaida.array(), _gradSaida.offset(),
            _gradEntrada.array(), _gradEntrada.offset(),
            _entrada.array(), _entrada.offset(),
            _entrada.tam()
        );

        return _gradEntrada;
    }

    private void elu(float[] x, int offX, float[] dest, int offDest, int n) {
        for (int i = 0; i < n; i++) {
            float val = x[offX + i];
            dest[offDest + i] = (val > 0) ? val : alpha * (float) (Math.exp(val) - 1.0);
        }
    }

    private void elud(float[] g, int offG, float[] gradE, int offGE, float[] x, int offX, int n) {
        for (int i = 0; i < n; i++) {
            final float grad = g[offG + i];
            final float entrada = x[offX + i];
            gradE[offGE + i] = grad * ((entrada > 0.0) ? 1.0f : alpha * (float) Math.exp(entrada));
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
