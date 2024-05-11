package jnn.avaliacao.metrica;

import jnn.core.Utils;
import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;

/**
 * Classe genérica para cálculos de métricas de avaliação dos modelos.
 * <p>
 *    Novas métricas devem implementar o método {@code calcular()}.
 * </p>
 */
abstract class Metrica {

	/**
	 * Utilitário.
	 */
	Utils utils = new Utils();

	/**
	 * Calcula a métrica de avaliação configurada.
	 * @param modelo modelo desejado.
	 * @param entrada {@code Tensores} com dados de entrada para o modelo.
	 * @param real {@code Tensores} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor calcular(Modelo modelo, Tensor[] entrada, Tensor[] saida) {
		throw new UnsupportedOperationException(
			"É necessário implementar a métrica de avaliação do modelo."
		);
	}

	/**
	 * Calcula a matriz de confusão de acordo com as previsões do modelo.
	 * @param modelo modelo desejado.
	 * @param entrada {@code Tensores} com dados de entrada para o modelo.
	 * @param real {@code Tensores} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor calcularMatriz(Modelo modelo, Tensor[] entrada, Tensor[] real) {
		throw new UnsupportedOperationException(
			"É necessário implementar a métrica de avaliação do modelo."
		);
	}

	/**
	 * <p>
	 *    Auxiliar.
	 * </p>
	 * Encontra o índice com o maior valor contido no array fornecido
	 * @param arr array contendo os dados
	 * @return índice com o maior valor contido nos dados.
	 */
	protected int indiceMaiorValor(double[] arr) {
		int maiorId = 0;
		double maiorVal = arr[0];
  
		for (int i = 1; i < arr.length; i++) {
			if (arr[i] > maiorVal) {
				maiorVal = arr[i];
				maiorId = i;
			}
		}
  
		return maiorId;
	}

	/**
	 * <p>
	 *    Auxiliar.
	 * </p>
	 * Encontra o índice com o maior valor contido no array fornecido
	 * @param tensor array contendo os dados
	 * @return índice com o maior valor contido nos dados.
	 */
	protected int indiceMaiorValor(Tensor tensor) {
		if (tensor.numDim() != 1) {
			throw new UnsupportedOperationException(
				"\nSem suporte para tensores com mais de uma dimensão."
			);
		}

		int maiorId = 0;
		int n = tensor.shape()[0];
		double maiorVal = tensor.get(0);
		for (int i = 1; i < n; i++) {
			if (tensor.get(i) > maiorVal) {
				maiorVal = tensor.get(i);
				maiorId = i;
			}
		}
  
		return maiorId;
	}

	/**
	 * <p>
	 *    Auxiliar.
	 * </p>
	 * Encontra o índice com o maior valor contido no array fornecido
	 * @param arr array contendo os dados
	 * @return índice com o maior valor contido nos dados.
	 */
	protected int indiceMaiorValor(Double[] arr) {
		int maiorId = 0;
		double maiorVal = arr[0];
  
		for (int i = 1; i < arr.length; i++) {
			if (arr[i] > maiorVal) {
				maiorVal = arr[i];
				maiorId = i;
			}
		}
  
		return maiorId;
	}

	/**
	 * <p>
	 *    Auxiliar.
	 * </p>
	 * Calcula a matriz de confusão.
	 * @param modelo modelo para avaliar.
	 * @param entradas conjunto de entradas.
	 * @param saidas conjunto de saídas.
	 * @return matríz de confusão calculada.
	 */
	protected Tensor matrizConfusao(Modelo modelo, Tensor[] entrada, Tensor[] saidas) {
		Tensor[] prevs = modelo.forwards(entrada);

		if (prevs[0].numDim() != 1) {
			throw new UnsupportedOperationException(
				"\nO modelo deve prever apenas dados de uma dimensão (arrays)."
			);
		}

		int nClasses = prevs[0].tamanho();
		Tensor matriz = new Tensor(nClasses, nClasses);

		for (int i = 0; i < prevs.length; i++) {
			int previsto = indiceMaiorValor(prevs[i]);
			int real = indiceMaiorValor(saidas[i]);
			double val = matriz.get(real, previsto);
			matriz.set((val += 1), real, previsto);
		}

		return matriz;
	} 
}
