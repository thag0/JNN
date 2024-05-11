package jnn.inicializadores;

import jnn.core.tensor.Tensor;

/**
 * Inizialiciador para tensores com valor zero.
 */
public class Zeros extends Inicializador {

	/**
	 * Inizialiciador para tensores com valor zero.
	 */
	public Zeros() {}

	@Override
	public void inicializar(Tensor tensor) {
		tensor.preencher(0);
	}

}
