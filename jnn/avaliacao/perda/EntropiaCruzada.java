package jnn.avaliacao.perda;

import jnn.core.tensor.Tensor;

/**
 * Função de perda Cross Entropy, que é normalmente usada em problemas 
 * de classificação multiclasse. Ela mede a discrepância entre a distribuição 
 * de probabilidade prevista e a distribuição de probabilidade real dos rótulos.
 */
public class EntropiaCruzada extends Perda {
	double eps = 1e-8;//evitar log 0

	/**
	 * Inicializa a função de perda Categorical Cross Entropy.
	 */
	public EntropiaCruzada() {}

	@Override
	public Tensor forward(Tensor prev, Tensor real) {
		super.verificarDimensoes(prev, real);
		int tam = prev.tam();
		
		double ec = 0.0;
		for (int i = 0; i < tam; i++) {
			ec += real.get(i) * Math.log(prev.get(i) + eps);
		}
		
		return new Tensor(new double[]{ -ec/tam }, 1);
	}
	
	@Override
	public Tensor backward(Tensor prev, Tensor real) {
		super.verificarDimensoes(prev, real);

		return prev.map(
			real,
			(p, r) -> (p - r)
		);
	}
}
