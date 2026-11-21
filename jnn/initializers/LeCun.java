package jnn.initializers;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

/**
 * Inicializador LeCun para uso dentro da biblioteca.
 */
public class LeCun extends Initializer {
	
	/**
	 * Instancia um inicializador LeCun para tensores com seed
	 * aleatória.
	 */
	public LeCun() {}

	@Override
	public void forward(Tensor tensor) {
        int fanIn = calculateFans(tensor)[0];
        float var = 1.0f / fanIn;

		tensor.apply(_ -> JNNutils.randGaussianf() * (float) Math.sqrt(var));
	}

}
