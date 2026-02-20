package jnn.inicializadores;

import jnn.core.JNNutils;
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

	@Override
	public void forward(Tensor tensor) {
		tensor.aplicar(_ -> JNNutils.randGaussianf());
	}
	
}
