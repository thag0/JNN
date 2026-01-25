package jnn.inicializadores;

import jnn.core.JNNutils;
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

	@Override
	public void forward(Tensor tensor) {
		int fanIn = calcularFans(tensor)[0];
		double desvP = Math.sqrt(2.0 / fanIn);

		tensor.aplicar(_ -> JNNutils.randGaussian() * desvP);
	}

}
