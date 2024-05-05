package jnn.inicializadores;

import jnn.core.tensor.Tensor4D;

/**
 * Inicializador Xavier para uso dentro da biblioteca.
 */
public class GlorotUniforme extends Inicializador {

	/**
	 * Instância um inicializador Xavier para matrizes com seed
	 * aleatória.
	 */
	public GlorotUniforme() {}
	
	/**
	 * Instância um inicializador Xavier para matrizes.
	 * @param seed seed usada pelo gerador de números aleatórios.
	 */
	public GlorotUniforme(long seed) {
		super(seed);
	}

	@Override
	public void inicializar(Tensor4D tensor) {
		double limite = Math.sqrt(2.0 / (tensor.dim3() + tensor.dim4()));

		tensor.aplicar(x -> super.random.nextDouble() * (2.0 * limite) - limite);
	}

	@Override
	public void inicializar(Tensor4D tensor, int dim1) {
		double limite = Math.sqrt(2.0 / (tensor.dim3() + tensor.dim4()));

		tensor.aplicar(dim1, 
			x ->  super.random.nextDouble() * (2.0 * limite) - limite
		);
	}

	@Override
	public void inicializar(Tensor4D tensor, int dim1, int dim2) {
		double limite = Math.sqrt(6.0 / (tensor.dim3() + tensor.dim4()));

		tensor.aplicar(dim1, dim2, 
			x ->  super.random.nextDouble() * (2.0 * limite) - limite
		);
	}

	@Override
	public void inicializar(Tensor4D tensor, int dim1, int dim2, int dim3) {
		double limite = Math.sqrt(6.0 / tensor.dim4());

		tensor.aplicar(dim1, dim2, dim3, 
			x ->  super.random.nextDouble() * (2.0 * limite) - limite
		);
	}
}
