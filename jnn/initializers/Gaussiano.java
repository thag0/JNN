package jnn.initializers;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

/**
 * Inicializador Gaussiano para uso dentro da biblioteca.
 */
public class Gaussiano extends Initializer {

	/**
	 * Instancia um inicializador Gaussiano para tensores com seed
	 * aleatória.
	 */
	public Gaussiano() {}

	@Override
	public void forward(Tensor tensor) {
		tensor.apply(_ -> JNNutils.randGaussianf());
	}
	
}
