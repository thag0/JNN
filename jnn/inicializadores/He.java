package jnn.inicializadores;

import jnn.core.tensor.Tensor;

/**
 * Inicializador He para uso dentro da biblioteca.
 */
public class He extends Inicializador {

	/**
	 * Instancia um inicializador He para tensores com seed
	 * aleatória.
	 */
	public He() {}

	/**
	 * Instancia um inicializador He para tensores com seed
	 * aleatória.
	 * @param seed seed usada pelo gerador de números aleatórios.
	 */
	public He(long seed) {
		super(seed);
	}

	@Override
	public void inicializar(Tensor tensor) {
		int fanIn = calcularFans(tensor)[0];
		double desvP = Math.sqrt(2.0 / fanIn);

		tensor.aplicar(x -> random.nextGaussian()*desvP);
	}

}
