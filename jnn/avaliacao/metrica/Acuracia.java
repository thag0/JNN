package jnn.avaliacao.metrica;

import jnn.core.tensor.Tensor;

/**
 * Acurácia para avaliação de modelos de classificação.
 */
public class Acuracia extends Metrica {

    /**
     * Instancia a métrica de <strong>Acurácia</strong>.
     * <p>
     *		A acurácia é a proporção de previsões corretas em relação ao 
	 *		total de exemplos avaliados. Ela é calculada como o número de 
	 *		previsões corretas dividido pelo número total de exemplos.
     * </p>
     */
	public Acuracia() {}

	@Override
	public Tensor forward(Tensor[] prev, Tensor[] real) {
		validarDados(prev, real);

		int n = prev.length;
		int acertos = 0;
		
		for (int i = 0; i < n; i++) {
			int idPrev = idMaiorValor(prev[i]);
			int idReal = idMaiorValor(real[i]);

			if (idPrev == idReal) acertos++;
		}

		double acc = (double)acertos / n;

		return new Tensor(new double[]{ acc }, 1);
	}
}
