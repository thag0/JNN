package jnn.initializers;

import jnn.core.tensor.Tensor;

/**
 * Inicializador de valor constante para uso dentro da biblioteca.
 */
public class Constante extends Initializer {
	
	/**
	 * Valor de preenchimento.
	 */
	private final float val;

	/**
	 * Instancia um inicializador de valor constante.
	 * @param x valor usado de constante na inicialização.
	 */
	public Constante(Number x) {
		val = x.floatValue();
	}

	/**
	 * Instancia um inicializador de valor constante.
	 * <p>
	 *    Por padrão o valor é zero.
	 * </p>
	 */
	public Constante() {
		this(0.0);
	}

	@Override
	public void forward(Tensor tensor) {
		tensor.fill(val);
	}
	
}
