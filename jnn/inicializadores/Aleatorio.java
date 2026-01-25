package jnn.inicializadores;

import jnn.core.JNNutils;
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
	 * Instancia um inicializador de valores aleatórios com seed
	 * também aleatória.
	 */
	public Aleatorio() {
		this(-1.0, 1.0);
	}

	@Override
	public void forward(Tensor tensor) {
		tensor.aplicar(_ -> JNNutils.randDouble(min, max));
	}
	
}
