package jnn.ativacoes;

import jnn.camadas.Conv2D;
import jnn.camadas.Densa;
import jnn.core.OpTensor;
import jnn.core.tensor.Tensor;

/**
 * Implementação da função de ativação Softmax para uso
 * dentro dos modelos.
 */
public class Softmax extends Ativacao {

	/**
	 * Operador para tensores.
	 */
	OpTensor opt = new OpTensor();

	/**
	 * Instancia a função de ativação Softmax.
	 * <p>
	 *    A função Softmax transforma os valores de entrada em probabilidades
	 *    normalizadas,
	 *    permitindo que a unidade com a maior saída tenha uma probabilidade mais
	 *    alta.
	 * </p>
	 * Exemplo:
	 * <pre>
	 * tensor = [
	 *    [1, 2, 3]
	 *]
	 *
	 *softmax.forward(tensor, tensor);
	 *
	 *tensor = [
	 *    [0.090031, 0.244728, 0.665241]
	 *]
	 * </pre>
	 * <p>
	 *		Para tensores com duas dimensões, a softmax é aplicada a cada
	 *		"linha" do tensor.
	 * </p>
	 * <p>
	 * 		Para tensores com mais de três dimensões (3D+), a função não
	 * 		possui suporte.
	 * </p>
	 */
	public Softmax() {}

	@Override
	public void forward(Tensor x, Tensor dest) {
		if (x.numDim() != dest.numDim()) {
			throw new IllegalArgumentException(
				"\nTamanho do tensor de entrada (" + x.numDim() + ") " +
				"deve ser igual ao tamanho do tensor de saída (" + dest.numDim() + ")"
			);
		}

		int[] shape = x.shape();
		if (shape.length == 1) {
			double somaExp = 0;
			int cols = shape[0];
			for (int i = 0; i < cols; i++) {
				somaExp += Math.exp(x.get(i));
			}
			for (int i = 0; i < cols; i++) {
				double s = Math.exp(x.get(i)) / somaExp;
				dest.set(s, i);
			}

		} else if (shape.length == 2) {
			int lin = shape[0];
			int col = shape[1];
	
			for (int i = 0; i < lin; i++) {
				double somaExp = 0;
				for (int j = 0; j < col; j++) {
					somaExp += Math.exp(x.get(i, j));
				}
				for (int j = 0; j < col; j++) {
					double s = Math.exp(x.get(i, j)) / somaExp;
					dest.set(s, i, j);
				}
			}

		} else {
			throw new UnsupportedOperationException(
				"\nSuporte apenas para tensores 1D e 2D."
			);
		}
	}

	@Override
	public void backward(Densa camada) {
		int n = camada._buffer.tam();
		Tensor tmp = camada.saida().bloco(n);
		Tensor ident = new Tensor(n, n);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				ident.set((i == j ? 1.0 : 0.0), i, j);
			}
		}

		Tensor transp = tmp.transpor();

		Tensor res = opt.matMul(
			camada._gradSaida, 
			opt.matHad(
				tmp,
				opt.matSub(
					ident,
					transp
				)
			)
		);

		camada._gradSaida.copiar(res);
	}

	@Override
	public void backward(Conv2D camada) {
		throw new UnsupportedOperationException(
			"\nSem suporte para derivada " + nome() + " em camadas convolucionais."
		);
	}

}
