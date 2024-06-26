package jnn.inicializadores;

import jnn.core.tensor.Tensor;

/**
 * Inicializador de valores aleatórios para uso dentro da biblioteca.
 */
public class Aleatorio extends Inicializador {

	/**
	 * Valor mínimo de aleatorização.
	 */
	private final double min;

	/**
	 * Valor máximo de aleatorização.
	 */
	private final double max;

	/**
	 * Instancia um inicializador de valores aleatórios com seed
	 * também aleatória.
	 * @param min valor mínimo de aleatorização.
	 * @param max valor máximo de aleatorização.
	 */
	public Aleatorio(double min, double max) {
		if (min >= max) {
			throw new IllegalArgumentException(
				"O valor mínimo (" + min +") " +
				"deve ser menor que o valor máximo (" + max + ")."
			);
		}

		this.min = min;
		this.max = max;
	}

	/**
	 * Instancia um inicializador de valores aleatórios.
	 * @param min valor mínimo de aleatorização.
	 * @param max valor máximo de aleatorização.
	 * @param seed seed usada pelo gerador de números aleatórios.
	 */
	public Aleatorio(double min, double max, long seed) {
		super(seed);

		if (min >= max) {
			throw new IllegalArgumentException(
				"O valor mínimo (" + min +") " +
				"deve ser menor que o valor máximo (" + max + ")."
			);
		}

		this.min = min;
		this.max = max;
	}

	/**
	 * Instancia um inicializador de valores aleatórios com seed
	 * também aleatória.
	 */
	public Aleatorio() {
		this(-1.0, 1.0);
	}

	/**
	 * Instancia um inicializador de valores aleatórios.
	 * @param seed seed usada pelo gerador de números aleatórios.
	 */
	public Aleatorio(long seed) {
		super(seed);
		this.min = -1;
		this.max =  1;
	}

	@Override
	public void inicializar(Tensor tensor) {
		tensor.aplicar(x -> super.random.nextDouble(min, max));
	}
	
}
