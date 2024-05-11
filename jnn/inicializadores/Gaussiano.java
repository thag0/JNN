package jnn.inicializadores;

import jnn.core.tensor.Tensor;

/**
 * Inicializador Gaussiano para uso dentro da biblioteca.
 */
public class Gaussiano extends Inicializador {

	/**
	 * Instância um inicializador Gaussiano para matrizes com seed
	 * aleatória.
	 */
	public Gaussiano() {}

	/**
	 * Instância um inicializador Gaussiano para matrizes.
	 * @param seed seed usada pelo gerador de números aleatórios.
	 */
	public Gaussiano(long seed) {
		super(seed);
	}

	@Override
	public void inicializar(Tensor tensor) {
		tensor.aplicar(x -> super.random.nextGaussian());
	}
	
}
