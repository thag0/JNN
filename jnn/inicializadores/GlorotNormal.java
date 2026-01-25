package jnn.inicializadores;

import jnn.core.JNNutils;
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

	@Override
	public void forward(Tensor tensor) {
		int[] fans = calcularFans(tensor);
		double sigma = Math.sqrt(2.0 / (fans[0] + fans[1]));

		tensor.aplicar(_ -> JNNutils.randGaussian() * sigma);
	}
}
