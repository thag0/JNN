package jnn.acts;

import jnn.core.tensor.Tensor;

/**
 * Implementação da função de ativação Argmax para uso 
 * dentro dos modelos.
 */
public class Argmax extends Ativacao {

	/**
	 * Intancia uma nova função de ativação Argmax.
	 * <p>
	 *    A função argmax encontra o maior valor de saída dentre os neurônios
	 *    da camada e converte ele para 1, as demais saídas dos neurônios serão
	 *    convertidas para zero, fazendo a camada classificar uma única saída com
	 *    base no maior valor.
	 * </p>
	 * Exemplo:
	 * <pre>
	 * tensor = [
	 *    [1, 2, 3]
	 *]
	 *
	 *argmax.forward(tensor, tensor);
	 *
	 *tensor = [
	 *    [0, 0, 1]
	 *]
	 * </pre>
	 */
	public Argmax() {}

	@Override
	public void forward(Tensor x, Tensor dest) {
		if (!x.compShape(dest)) {
			throw new IllegalArgumentException(
				"\nTensor de entrada " + x.shapeStr() + " deve ter " +
				"mesmo formato do tensor de saída " + dest.shapeStr()
			);
		}

		if (x.numDim() > 1) {
			throw new UnsupportedOperationException(
				"\nSuporte apenas para tensores 1D."
			);
		}

		int cols = x.tam();
		int maxId = 0;
		double maxVal = x.get(0);

		for (int i = 1; i < cols; i++) {
			if (x.get(i) > maxVal) {
				maxId = i;
				maxVal = x.get(i);
			}
		}

		for (int i = 0; i < cols; i++) {
			dest.set(((i == maxId) ? 1.0 : 0.0), i);
		}
	}

	@Override
	public void backward(Tensor x, Tensor grad, Tensor dest) {
		throw new UnsupportedOperationException(
			"\nArgmax não possui derivada."
		);
	}
}
