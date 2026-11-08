package jnn.metrics.metric;

import jnn.core.tensor.Tensor;

/**
 * Classe genérica para cálculos de métricas de avaliação dos modelos.
 * <p>
 *    Novas métricas devem implementar o método {@code calcular()}.
 * </p>
 */
abstract class Metric {

	/**
	 * Verifica se os dados previstos e reais possuem a mesma quantidade
	 * de amostras.
	 * @param p {@code Tensores} com dados previstos.
	 * @param r {@code Tensores} com dados reais (rótulos).
	 */
	protected void validadeData(Tensor[] p, Tensor[] r) {
		if (p.length != r.length) {
			throw new IllegalArgumentException(
				"\nQuantidade de amostras preditas (" + p.length + ") " +
				"e rótulos (" + r.length + ") devem ser iguais."
			);
		}
	}

	/**
	 * Calcula a métrica de avaliação configurada.
	 * @param prev {@code Tensores} com dados previstos.
	 * @param real {@code Tensores} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public abstract Tensor forward(Tensor[] prev, Tensor[] real);

	/**
	 * <p>
	 *    Auxiliar.
	 * </p>
	 * Encontra o índice com o maior valor contido tensor fornecido.
	 * @param tensor tensor desejado.
	 * @return índice com o maior valor.
	 */
	protected int idMaxVal(Tensor tensor) {
		if (tensor.numDim() != 1) {
			throw new UnsupportedOperationException(
				"\nSem suporte para tensores com mais de uma dimensão."
			);
		}

		int idMax = 0;
		double maxVal = tensor.get(0);
		int n = tensor.shape()[0];
		for (int i = 1; i < n; i++) {
			if (tensor.get(i) > maxVal) {
				maxVal = tensor.get(i);
				idMax = i;
			}
		}
  
		return idMax;
	}

	/**
	 * <p>
	 *    Auxiliar.
	 * </p>
	 * Calcula a matriz de confusão.
	 * @param prev {@code Tensores} com dados previstos.
	 * @param real {@code Tensores} com dados reais.
	 * @return {@code Tensor} contendo a matriz de confusão no formato (real, previsto).
	 */
	protected Tensor confusionMatrix(Tensor[] prev, Tensor[] real) {
		if (prev[0].numDim() != 1 || real[0].numDim() != 1) {
			throw new UnsupportedOperationException(
				"\nSuporte apenas parra tensores 1D."
			);
		}

		int numClasses = prev[0].len();
		Tensor cm = new Tensor(numClasses, numClasses);

		for (int i = 0; i < prev.length; i++) {
			int p = idMaxVal(prev[i]);
			int r = idMaxVal(real[i]);
			double val = cm.get(r, p);
			cm.set((val += 1.0), r, p);
		}

		return cm;
	} 
}
