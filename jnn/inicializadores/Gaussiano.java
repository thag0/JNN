package jnn.inicializadores;

import jnn.core.tensor.Tensor;

/**
 * Inicializador Gaussiano para uso dentro da biblioteca.
 */
public class Gaussiano extends Inicializador {

	/**
	 * Instancia um inicializador Gaussiano para tensores com seed
	 * aleatória.
	 */
	public Gaussiano() {}

	/**
	 * Instancia um inicializador Gaussiano para tensores.
	 * @param seed seed usada pelo gerador de números aleatórios.
	 */
	public Gaussiano(Number seed) {
		setSeed(seed);
	}

	@Override
	public void forward(Tensor tensor) {
		tensor.aplicar(_ -> super.random.nextGaussian());
	}
	
}
