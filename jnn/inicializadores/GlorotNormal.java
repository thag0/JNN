package jnn.inicializadores;

import jnn.core.tensor.Tensor4D;

/**
 * Inicializador Glorot normalizado para uso dentro da biblioteca.
 */
public class GlorotNormal extends Inicializador {

	/**
	 * Instância um inicializador Glorot normalizado para matrizes 
	 * com seed
	 * aleatória.
	 */
	public GlorotNormal() {}

	/**
	 * Instância um inicializador Glorot normalizado para matrizes.
	 * @param seed seed usada pelo gerador de números aleatórios.
	 */
	public GlorotNormal(long seed) {
		super(seed);
	}

	@Override
	public void inicializar(Tensor4D tensor) {
		double desvio = Math.sqrt(2.0 / (tensor.dim3() + tensor.dim4()));

		tensor.aplicar(x ->  super.random.nextGaussian() * desvio);
	}

	@Override
	public void inicializar(Tensor4D tensor, int dim1) {
		double desvio = Math.sqrt(2.0 / (tensor.dim3() + tensor.dim4()));

		tensor.aplicar(dim1, 
			x -> super.random.nextGaussian() * desvio
		);
	}

	@Override
	public void inicializar(Tensor4D tensor, int dim1, int dim2) {
		double desvio = Math.sqrt(2.0 / (tensor.dim3() + tensor.dim4()));

		tensor.aplicar(dim1, dim2,
			x -> super.random.nextGaussian() * desvio
		);
	}

	@Override
	public void inicializar(Tensor4D tensor, int dim1, int dim2, int dim3) {
		double desvio = Math.sqrt(2.0 / tensor.dim4());

		tensor.aplicar(dim1, dim2, dim3,
			x -> super.random.nextGaussian() * desvio
		);
	}
}
