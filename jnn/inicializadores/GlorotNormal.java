package jnn.inicializadores;

import jnn.core.tensor.Tensor;

/**
 * Inicializador Glorot normalizado para uso dentro da biblioteca.
 */
public class GlorotNormal extends Inicializador {

	/**
	 * Instância um inicializador Glorot normalizado para matrizes 
	 * com seed
	 * aleatória.
	 */
	public GlorotNormal() {}

	/**
	 * Instância um inicializador Glorot normalizado para matrizes.
	 * @param seed seed usada pelo gerador de números aleatórios.
	 */
	public GlorotNormal(long seed) {
		super(seed);
	}

	@Override
	public void inicializar(Tensor tensor) {
		int[] fans = calcularFans(tensor);
		double sigma = Math.sqrt(2.0 / (fans[0] + fans[1]));

		tensor.aplicar(x -> random.nextGaussian() * sigma);
	}
}
