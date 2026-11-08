package jnn.metrics.metric;

import jnn.core.tensor.Tensor;

/**
 * Acurácia para avaliação de modelos de classificação.
 */
public class Accuracy extends Metric {

    /**
     * Instancia a métrica de <strong>Acurácia</strong>.
     * <p>
     *		A acurácia é a proporção de previsões corretas em relação ao 
	 *		total de exemplos avaliados. Ela é calculada como o número de 
	 *		previsões corretas dividido pelo número total de exemplos.
     * </p>
     */
	public Accuracy() {}

	@Override
	public Tensor forward(Tensor[] prev, Tensor[] real) {
		validadeData(prev, real);

		int hits = 0;
		int len = prev.length;
		
		for (int i = 0; i < len; i++) {
			int idPrev = idMaxVal(prev[i]);
			int idReal = idMaxVal(real[i]);

			if (idPrev == idReal) hits++;
		}

		float acc = (float) hits / len;

		return new Tensor(1).set(acc, 0);
	}
}
