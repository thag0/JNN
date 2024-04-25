package jnn.avaliacao.metrica;

import jnn.core.Utils;
import jnn.modelos.Modelo;

/**
 * Classe genérica para cálculos de métricas de avaliação da rede neural.
 * <p>
 *    Novas métricas devem implementar o método {@code calcular()}.
 * </p>
 */
abstract class Metrica {

	/**
	 * Transofar entradas em arrays por enquanto.
	 */
	Utils utils = new Utils();

	/**
	 * Calcula a métrica de avaliação configurada.
	 * @param rede rede neural.
	 * @param entrada dados de entrada.
	 * @param saida dados de saída relativos a entrada.
	 * @return valor de avaliação de acordo com a métrica configurada
	 */
	public double calcular(Modelo rede, Object entrada, Object[] saida) {
		throw new UnsupportedOperationException(
			"É necessário implementar a métrica de avaliação da rede."
		);
	}

	/**
	 * Calcula a métrica de avaliação configurada.
	 * @param rede rede neural.
	 * @param entrada dados de entrada.
	 * @param saida dados de saída relativos a entrada.
	 * @return valor de avaliação de acordo com a métrica configurada
	 */
	public int[][] calcularMatriz(Modelo rede, Object entrada, double[][] saida) {
		throw new UnsupportedOperationException(
			"É necessário implementar a métrica de avaliação da rede."
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
	 * Calcula a matriz de confusão.
	 * @param modelo
	 * @param entradas
	 * @param saidas
	 * @return
	 */
	protected int[][] matrizConfusao(Modelo modelo, Object entradas, Object[] saidas) {
		if (!(saidas instanceof double[][])) {
			throw new IllegalArgumentException(
				"Objeto esperado para saída é double[][], recebido " + saidas.getClass().getTypeName()
			);
		}

		Object[] arrEntrada = utils.transformarParaArray(entradas);

		double[][] s = (double[][]) saidas;

		int nClasses = s[0].length;
		int[][] matriz = new int[nClasses][nClasses];
		double[] saidaRede = new double[modelo.camadaSaida().tamanhoSaida()];

		for (int i = 0; i < arrEntrada.length; i++) {
			modelo.forward(arrEntrada[i]);
			saidaRede = modelo.saidaParaArray();

			int real = this.indiceMaiorValor(s[i]);
			int previsto = this.indiceMaiorValor(saidaRede);

			matriz[real][previsto]++;
		}

		return matriz;
	} 
}
