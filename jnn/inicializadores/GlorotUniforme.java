package jnn.inicializadores;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

/**
 * Inicializador Xavier para uso dentro da biblioteca.
 */
public class GlorotUniforme extends Inicializador {

	/**
	 * Instancia um inicializador Xavier para tensores com seed
	 * aleatória.
	 */
	public GlorotUniforme() {}

	@Override
	public void forward(Tensor tensor) {
		int[] fans = calcularFans(tensor);

		int fin  = fans[0];
		int fout = fans[1];
		float limite = (float) Math.sqrt(6.0 / (fin + fout));

		tensor.apply(_ -> JNNutils.randFloat() * (2.0f * limite) - limite);
	}
}
