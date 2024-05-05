package jnn.inicializadores;

import jnn.core.tensor.Tensor4D;

/**
 * Inicializador He para uso dentro da biblioteca.
 */
public class He extends Inicializador {

	/**
	 * Instância um inicializador He para matrizes com seed
	 * aleatória.
	 */
	public He() {}

	@Override
	public void inicializar(Tensor4D tensor) {
		double a = Math.sqrt(2.0 / tensor.dim3());

		tensor.aplicar(x -> a * super.random.nextGaussian());
	}

	@Override
	public void inicializar(Tensor4D tensor, int dim1) {
		double a = Math.sqrt(2.0 / tensor.dim3());
		
		tensor.aplicar(dim1, 
			x ->  a * super.random.nextGaussian()
		);
	}

	@Override
	public void inicializar(Tensor4D tensor, int dim1, int dim2) {
		double a = Math.sqrt(2.0 / tensor.dim3());
		
		tensor.aplicar(dim1, dim2, 
			x ->  a * super.random.nextGaussian()
		);
	}

	@Override
	public void inicializar(Tensor4D tensor, int dim1, int dim2, int dim3) {
		double a = Math.sqrt(2.0 / tensor.dim3());
		
		tensor.aplicar(dim1, dim2, dim3, 
			x ->  a * super.random.nextGaussian()
		);
	}
}
