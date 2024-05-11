package jnn.inicializadores;

import jnn.core.tensor.Tensor;

/**
 * Inicializador LeCun para uso dentro da biblioteca.
 */
public class LeCun extends Inicializador {
	
	/**
	 * Instância um inicializador LeCun para matrizes com seed
	 * aleatória.
	 */
	public LeCun() {}
	
	/**
	 * Instância um inicializador LeCun para matrizes.
	 * @param seed seed usada pelo gerador de números aleatórios.
	 */
	public LeCun(long seed) {
		super(seed);
	}

	@Override
	public void inicializar(Tensor tensor) {
        int fanIn = calcularFans(tensor)[0];
        double var = 1.0 / fanIn;

		tensor.aplicar(x -> random.nextGaussian() * Math.sqrt(var));
	}

}
