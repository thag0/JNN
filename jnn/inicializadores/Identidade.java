package jnn.inicializadores;

import jnn.core.tensor.Tensor;

/**
 * Inicializador de matriz identidade para uso dentro da biblioteca.
 */
public class Identidade extends Inicializador {

	/**
	 * Instância um inicializador de matriz identidade.
	 */
	public Identidade() {}

	@Override
	public void inicializar(Tensor tensor) {
		if (tensor.numDim() < 2) {
			throw new UnsupportedOperationException(
				"\nIncialização identidade só é aplicada em tensores 2D."
			);
		}

		int[] shape = tensor.shape();
		int lin = shape[0];
		int col = shape[1];
		
		for (int i = 0; i < lin; i++) {
			for (int j = 0; j < col; j++) {
				tensor.set((i == j ? 1.0 : 0.0), i, j);
			}
		}
	
	}

}
