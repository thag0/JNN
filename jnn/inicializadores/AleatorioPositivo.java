package jnn.inicializadores;

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
	 * Instancia um inicializador de valores aleatórios positivos.
	 * @param max valor máximo de aleatorização.
	 * @param seed seed usada pelo gerador de números aleatórios.
	 */
	public AleatorioPositivo(Number max, Number seed) {
		this(max);
		setSeed(seed);
	}

	/**
	 * Instancia um inicializador de valores aleatórios positivos
	 * com seed aleatória.
	 */
	public AleatorioPositivo() {
		this(1.0);
	}

	/**
	 * Instancia um inicializador de valores aleatórios positivos.
	 * @param seed seed usada pelo gerador de números aleatórios.
	 */
	public AleatorioPositivo(long seed) {
		this(1.0, seed);
	}

	@Override
	public void forward(Tensor tensor) {
		tensor.aplicar(x -> super.random.nextDouble(0, max));
	}
	
}
