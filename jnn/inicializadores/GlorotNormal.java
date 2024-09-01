package jnn.inicializadores;

import jnn.core.tensor.Tensor;

/**
 * Inicializador Glorot normalizado para uso dentro da biblioteca.
 */
public class GlorotNormal extends Inicializador {

	/**
	 * Instancia um inicializador Glorot normalizado para tensores 
	 * com seed aleatória.
	 */
	public GlorotNormal() {}

	/**
	 * Instancia um inicializador Glorot normalizado para tensores.
	 * @param seed seed usada pelo gerador de números aleatórios.
	 */
	public GlorotNormal(Number seed) {
		setSeed(seed);
	}

	@Override
	public void forward(Tensor tensor) {
		int[] fans = calcularFans(tensor);
		double sigma = Math.sqrt(2.0 / (fans[0] + fans[1]));

		tensor.aplicar(x -> random.nextGaussian() * sigma);
	}
}
