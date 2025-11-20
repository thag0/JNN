package jnn.metrica.perda;

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
		verificarDimensoes(prev, real);
		
		if (prev.numDim() == 1) {
			int tam = prev.tam();
			double ec = f(prev, real, tam);
			
			return new Tensor(
				new double[]{ -ec/tam }
			);
		}

		int lotes = prev.tamDim(0);
		int amostras = prev.tamDim(1);

		double somaLote = 0;
		for (int i = 0; i < lotes; i++) {
			Tensor p = prev.subTensor(i);
			Tensor r = real.subTensor(i);

			double ec = f(p, r, amostras);
			somaLote += -ec;
		}

		return new Tensor(
			new double[] {somaLote / lotes}
		);
	}

	/**
	 * Cálculo interno da Cross Entropy.
	 * @param prev tensor com dados previstos.
	 * @param real tensor com dados reais.
	 * @param tam quantidade de amostras.
	 * @return soma do CE.
	 */
	private double f(Tensor prev, Tensor real, int tam) {
		double ec = 0.0;
		for (int i = 0; i < tam; i++) {
			ec += real.get(i) * Math.log(prev.get(i) + eps);
		}

		return ec;
	}
	
	@Override
	public Tensor backward(Tensor prev, Tensor real) {
		verificarDimensoes(prev, real);

		return prev.map(real,
			(p, r) -> (p - r)
		);
	}
}
