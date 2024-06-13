package jnn.inicializadores;

import jnn.core.tensor.Tensor;

/**
 * Inicializador de valor constante para uso dentro da biblioteca.
 */
public class Constante extends Inicializador {
	
	/**
	 * Valor de preenchimento.
	 */
	private final double val;

	/**
	 * Instancia um inicializador de valor constante.
	 * @param val valor usado de constante na inicialização.
	 */
	public Constante(double val) {
		this.val = val;
	}

	/**
	 * Instancia um inicializador de valor constante.
	 * <p>
	 *    Por padrão o valor é zero.
	 * </p>
	 */
	public Constante() {
		this(0.0d);
	}

	@Override
	public void inicializar(Tensor tensor) {
		tensor.aplicar(x -> val);
	}
	
}
