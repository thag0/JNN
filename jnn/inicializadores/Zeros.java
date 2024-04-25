package jnn.inicializadores;

import jnn.core.Tensor4D;

public class Zeros extends Inicializador {

	/**
	 * Inizialiciador para matrizes com valor zero.
	 */
	public Zeros() {}

	@Override
	public void inicializar(Tensor4D tensor) {
		tensor.preencher(0);
	}

	@Override
	public void inicializar(Tensor4D tensor, int dim1) {
		tensor.preencher3D(dim1, 0);
	}

	@Override
	public void inicializar(Tensor4D tensor, int dim1, int dim2) {
		tensor.preencher2D(dim1, dim2, 0);
	}

	@Override
	public void inicializar(Tensor4D tensor, int dim1, int dim2, int dim3) {
		tensor.preencher1D(dim1, dim2, dim3, 0);
	}
}
