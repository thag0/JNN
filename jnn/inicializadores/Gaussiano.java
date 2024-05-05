package jnn.inicializadores;

import jnn.core.tensor.Tensor4D;

/**
 * Inicializador Gaussiano para uso dentro da biblioteca.
 */
public class Gaussiano extends Inicializador {

	/**
	 * Instância um inicializador Gaussiano para matrizes com seed
	 * aleatória.
	 */
	public Gaussiano() {}

	/**
	 * Instância um inicializador Gaussiano para matrizes.
	 * @param seed seed usada pelo gerador de números aleatórios.
	 */
	public Gaussiano(long seed) {
		super(seed);
	}

	@Override
	public void inicializar(Tensor4D tensor) {
		tensor.aplicar(x -> super.random.nextGaussian());
	}

	@Override
	public void inicializar(Tensor4D tensor, int dim1) {
		tensor.aplicar(dim1, 
			x -> super.random.nextGaussian()
		);
	}

	@Override
	public void inicializar(Tensor4D tensor, int dim1, int dim2) {
		tensor.aplicar(dim1, dim2, 
			x ->  super.random.nextGaussian()
		);
	}

	@Override
	public void inicializar(Tensor4D tensor, int dim1, int dim2, int dim3) {
		tensor.aplicar(dim1, dim2, dim3, 
			x ->  super.random.nextGaussian()
		);
	}
}
