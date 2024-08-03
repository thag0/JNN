package jnn.avaliacao.perda;

import jnn.core.tensor.Tensor;

/**
 * Função de perda Binary Cross Entropy, que é usada em problemas de 
 * classificação binária. Ela é comumente aplicada quando há apenas duas 
 * classes possíveis.
 */
 public class EntropiaCruzadaBinaria extends Perda {
	double eps = 1e-7;

	/**
	 * Inicializa a função de perda Binary Cross Entropy.
	 */
	public EntropiaCruzadaBinaria() {}

	@Override
	public Tensor calcular(Tensor prev, Tensor real) {
		super.verificarDimensoes(prev, real);
		int tam = prev.tam();

		double ecb = 0;
		double p, r;
		for (int i = 0; i < tam; i++) {
			p = prev.get(i);
			r = real.get(i);
			ecb += (r * Math.log(p + eps)) + (1 - r) * (Math.log(1 - p + eps));
		}

		return new Tensor(new double[]{ (-ecb/tam) });
	}

	@Override
	public Tensor derivada(Tensor prev, Tensor real) {
		super.verificarDimensoes(prev, real);
		final int tam = prev.tam();

		return prev.map(
			real,
			(p, r) -> (((1.0 - r) / (1.0 - p)) - (r / p)) / tam
		);
	}
}
