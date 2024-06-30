package jnn.ativacoes;

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
	 *argmax.calcular(tensor, tensor);
	 *
	 *tensor = [
	 *    [0, 0, 1]
	 *]
	 * </pre>
	 */
	public Argmax() {}

	@Override
	public void forward(Tensor entrada, Tensor saida) {
		if (!entrada.compararShape(saida)) {
			throw new IllegalArgumentException(
				"\nTensor de entrada " + entrada.shapeStr() + " deve ter " +
				"mesmo formato do tensor de saída " + saida.shapeStr()
			);
		}

		if (entrada.numDim() > 1) {
			throw new UnsupportedOperationException(
				"\nSem suporte para tensores com mais de uma dimensão."
			);
		}

		int cols = entrada.tamanho();
		double maxId = 0;
		double maxVal = entrada.get(0);

		for (int i = 1; i < cols; i++) {
			if (entrada.get(i) > maxVal) {
				maxId = i;
				maxVal = entrada.get(i);
			}
		}

		for (int i = 0; i < cols; i++) {
			saida.set(((i == maxId) ? 1.0d : 0.0d), i);
		}
	}

	@Override
	public void backward(Tensor a, Tensor b, Tensor c) {
		throw new UnsupportedOperationException(
			"\nArgmax não possui derivada."
		);
	}
}
