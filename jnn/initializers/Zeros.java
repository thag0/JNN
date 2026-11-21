package jnn.initializers;

import jnn.core.tensor.Tensor;

/**
 * Inicializador para tensores com valor zero.
 */
public class Zeros extends Initializer {

	/**
	 * Inizialiciador para tensores com valor zero.
	 */
	public Zeros() {}

	@Override
	public void forward(Tensor tensor) {
		tensor.zero();
	}

}
