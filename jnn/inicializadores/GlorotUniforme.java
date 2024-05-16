package jnn.inicializadores;

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
	
	/**
	 * Instancia um inicializador Xavier para tensores.
	 * @param seed seed usada pelo gerador de números aleatórios.
	 */
	public GlorotUniforme(long seed) {
		super(seed);
	}

	@Override
	public void inicializar(Tensor tensor) {
		int[] fans = calcularFans(tensor);

		int fin  = fans[0];
		int fout = fans[1];
		double limite = Math.sqrt(2.0 / (fin + fout));

		tensor.aplicar(x -> super.random.nextDouble() * (2.0 * limite) - limite);
	}
}
