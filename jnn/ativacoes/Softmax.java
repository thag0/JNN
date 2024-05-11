package jnn.ativacoes;

import jnn.camadas.Convolucional;
import jnn.camadas.Densa;
import jnn.core.tensor.OpTensor;
import jnn.core.tensor.Tensor;

/**
 * Implementação da função de ativação Softmax para uso
 * dentro dos modelos.
 */
public class Softmax extends Ativacao {

	/**
	 * Operador para tensores.
	 */
	OpTensor optensor = new OpTensor();

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
	 *    [1, 2, 3],
	 *]
	 *
	 *softmax.forward(tensor, tensor);
	 *
	 *tensor = [
	 *    [0,090031, 0,244728, 0,665241],
	 *]
	 * </pre>
	 */
	public Softmax() {}

	@Override
	public void forward(Tensor entrada, Tensor saida) {
		if (entrada.numDim() != saida.numDim()) {
			throw new IllegalArgumentException(
				"\nTamanho do tensor de entrada (" + entrada.numDim() + ") " +
				"deve ser igual ao tamanho do tensor de saída (" + saida.numDim() + ")"
			);
		}

		int dims = entrada.numDim();
		if (dims > 1) {
			throw new UnsupportedOperationException(
				"\nSem suporte para tensores com mais de uma dimensão."
			);
		}

		double somaExp = 0;
		int cols = entrada.shape()[0];
		for (int i = 0; i < cols; i++) {
			somaExp += Math.exp(entrada.get(i));
		}
		for (int i = 0; i < cols; i++) {
			double s = Math.exp(entrada.get(i)) / somaExp;
			saida.set(s, i);
		}
	}

	@Override
	public void backward(Densa camada) {
		int n = camada._somatorio.tamanho();
		Tensor tmp = camada.saida().bloco(n);
		Tensor ident = new Tensor(n, n);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				ident.set((i == j ? 1 : 0), i, j);
			}
		}

		Tensor transp = tmp.transpor();

		Tensor res = optensor.matMult(
			camada._gradSaida, 
			optensor.matHadamard(
				tmp, optensor.matSub(ident, transp)
			)
		);

		camada._gradSaida.copiar(res);
	}

	@Override
	public void backward(Convolucional camada) {
		throw new UnsupportedOperationException(
			"\nSem suporte para derivada " + nome() + " em camadas convolucionais."
		);
	}

}
