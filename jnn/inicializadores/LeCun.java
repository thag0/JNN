package jnn.inicializadores;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

/**
 * Inicializador LeCun para uso dentro da biblioteca.
 */
public class LeCun extends Inicializador {
	
	/**
	 * Instancia um inicializador LeCun para tensores com seed
	 * aleatória.
	 */
	public LeCun() {}

	@Override
	public void forward(Tensor tensor) {
        int fanIn = calcularFans(tensor)[0];
        float var = 1.0f / fanIn;

		tensor.aplicar(_ -> JNNutils.randGaussianf() * (float) Math.sqrt(var));
	}

}
