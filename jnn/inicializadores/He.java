package jnn.inicializadores;

import jnn.core.tensor.Tensor;

/**
 * Inicializador He para uso dentro da biblioteca.
 */
public class He extends Inicializador {

	/**
	 * Instância um inicializador He para matrizes com seed
	 * aleatória.
	 */
	public He() {}

	@Override
	public void inicializar(Tensor tensor) {
		int fanIn = calcularFans(tensor)[0];
		double desvP = Math.sqrt(2.0 / fanIn);

		tensor.aplicar(x -> random.nextGaussian()*desvP);
	}

}
