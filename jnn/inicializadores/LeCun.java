package jnn.inicializadores;

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
	
	/**
	 * Instancia um inicializador LeCun para tensores.
	 * @param seed seed usada pelo gerador de números aleatórios.
	 */
	public LeCun(Number seed) {
		setSeed(seed);
	}

	@Override
	public void inicializar(Tensor tensor) {
        int fanIn = calcularFans(tensor)[0];
        double var = 1.0 / fanIn;

		tensor.aplicar(x -> random.nextGaussian() * Math.sqrt(var));
	}

}
