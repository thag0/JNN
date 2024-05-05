package jnn.ativacoes;

import jnn.camadas.Convolucional;
import jnn.camadas.Densa;
import jnn.core.tensor.OpTensor4D;
import jnn.core.tensor.Tensor4D;

/**
 * Implementação da função de ativação Softmax para uso
 * dentro dos modelos.
 */
public class Softmax extends Ativacao {

	/**
	 * Operador para tensores.
	 */
	OpTensor4D optensor = new OpTensor4D();

	/**
	 * Instancia a função de ativação Softmax.
	 * <p>
	 *    A função Softmax transforma os valores de entrada em probabilidades
	 *    normalizadas,
	 *    permitindo que a unidade com a maior saída tenha uma probabilidade mais
	 *    alta.
	 * </p>
	 * <p>
	 *    A ativação atua em cada linha do tensor individualmente
	 * </p>
	 * Exemplo:
	 * <pre>
	 * tensor = [[[
	 *    [1, 2, 3], 
	 *    [2, 3, 1], 
	 *    [3, 1, 2], 
	 *]]]
	 *
	 *softmax.calcular(tensor, tensor);
	 *
	 *tensor = [[[
	 *    [0.1, 0.2, 0.7], 
	 *    [0.2, 0.7, 0.1], 
	 *    [0.7, 0.1, 0.2], 
	 *]]]
	 * </pre>
	 */
	public Softmax() {}

	@Override
	public void forward(Tensor4D entrada, Tensor4D saida) {
		int lote = entrada.dim1();
		int canais = entrada.dim2();
		int linhas = entrada.dim3();
		int colunas = entrada.dim4();
	
		for (int l = 0; l < lote; l++) {
			for (int c = 0; c < canais; c++) {
				for (int lin = 0; lin < linhas; lin++) {
					double somaExp = 0;

					for (int col = 0; col < colunas; col++) {
						somaExp += Math.exp(entrada.get(l, c, lin, col));
					}

					for (int col = 0; col < colunas; col++) {
						double s = Math.exp(entrada.get(l, c, lin, col)) / somaExp;
						saida.set(s, l, c, lin, col);
					}
				}
			}
		}
		
	}

	@Override
	public void backward(Densa camada) {
		int n = camada._somatorio.dim4();
		Tensor4D tmp = camada.saida().bloco(0, 0, 0, n);
		Tensor4D ident = new Tensor4D(1, 1, n, camada._somatorio.dim4());
		ident.identidade(0, 0);
		Tensor4D transp = optensor.matTranspor(tmp, 0, 0);

		optensor.matMult(
			camada._gradSaida, 
			optensor.matHadamard(
				tmp,
				optensor.matSub(ident, transp, 0),
				0, 
				0
			), 
			camada._gradSaida
		);
	}

	@Override
	public void backward(Convolucional camada) {
		throw new UnsupportedOperationException(
			"\nSem suporte para derivada " + nome() + " em camadas convolucionais."
		);
	}

}
