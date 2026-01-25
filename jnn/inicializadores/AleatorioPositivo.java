package jnn.inicializadores;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

/**
 * Inicializador de valores aleatórios positivos para uso dentro da biblioteca.
 */
public class AleatorioPositivo extends Inicializador {

	/**
	 * Valor máximo de aleatorização.
	 */
	private final double max;

	/**
	 * Instancia um inicializador de valores aleatórios positivos
	 * com seed aleatória.
	 * @param max valor máximo de aleatorização.
	 */
	public AleatorioPositivo(Number max) {
		double m = max.doubleValue();

		if (m <= 0) {
			throw new IllegalArgumentException(
				"\nValor máximo deve ser maior que zero."
			);
		}

		this.max = m;
	}

	/**
	 * Instancia um inicializador de valores aleatórios positivos
	 * com seed aleatória.
	 */
	public AleatorioPositivo() {
		this(1.0);
	}

	@Override
	public void forward(Tensor tensor) {
		tensor.aplicar(_ -> JNNutils.randDouble(0, max));
	}
	
}
