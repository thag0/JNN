package jnn.avaliacao.metrica;

import jnn.core.tensor.Tensor;

/**
 * TODO
 */
public class Acuracia extends Metrica {

	/**
	 * TODO
	 */
	public Acuracia() {}

	@Override
	public Tensor calcular(Tensor[] prev, Tensor[] real) {
		validarDados(prev, real);

		int n = prev.length;
		int acertos = 0;
		
		for (int i = 0; i < n; i++) {
			int idPrev = indiceMaiorValor(prev[i]);
			int idReal = indiceMaiorValor(real[i]);

			if (idPrev == idReal) acertos++;
		}

		double acc = (double)acertos / n;

		return new Tensor(new double[]{ acc }, 1);
	}
}
