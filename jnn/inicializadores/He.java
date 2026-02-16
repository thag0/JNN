package jnn.inicializadores;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

/**
 * Inicializador He (Kaiming).
 *<p>
 *		Indicado para modelos com funções de ativação ReLU e variações, 
 *		pois mantém a variância dos gradientes estável ao longo das camadas, 
 *		reduzindo o risco de vanishing/exploding gradients.
 *</p>
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
		float desvP = (float) Math.sqrt(2.0 / fanIn);

		tensor.aplicar(_ -> JNNutils.randGaussianf() * desvP);
	}

}
