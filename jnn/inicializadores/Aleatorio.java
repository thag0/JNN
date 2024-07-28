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
	public Aleatorio(Number min, Number max) {
		double mi = min.doubleValue();
		double ma = max.doubleValue();
		if (mi >= ma) {
			throw new IllegalArgumentException(
				"\nValor mínimo (" + min +") " +
				"deve ser menor que o valor máximo (" + max + ")."
			);
		}

		this.min = mi;
		this.max = ma;
	}

	/**
	 * Instancia um inicializador de valores aleatórios.
	 * @param min valor mínimo de aleatorização.
	 * @param max valor máximo de aleatorização.
	 * @param seed seed usada pelo gerador de números aleatórios.
	 */
	public Aleatorio(Number min, Number max, Number seed) {
		this(min, max);
		setSeed(seed);
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
	public Aleatorio(Number seed) {
		this(-1.0, 1.0, seed);
	}

	@Override
	public void inicializar(Tensor tensor) {
		tensor.aplicar(x -> super.random.nextDouble(min, max));
	}
	
}
