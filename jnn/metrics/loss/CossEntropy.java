package jnn.metrics.loss;

import jnn.core.tensor.Tensor;

/**
 * Função de perda Cross Entropy, que é normalmente usada em problemas 
 * de classificação multiclasse. Ela mede a discrepância entre a distribuição 
 * de probabilidade prevista e a distribuição de probabilidade real dos rótulos.
 */
public class CossEntropy extends Loss {

	/**
	 * Pequeno valor para evitar log de zero.
	 */
	float eps = 1e-7f;

	/**
	 * Inicializa a função de perda Categorical Cross Entropy.
	 */
	public CossEntropy() {}

	@Override
	public Tensor forward(Tensor prev, Tensor real) {
		validateDimensions(prev, real);
		
		if (prev.numDim() == 1) {
			int len = prev.len();
			float ce = f(prev, real, len);
			
			return new Tensor(1).set((-ce/len), 0);
		}

		int batches = prev.lenDim(0);
		int samples = prev.lenDim(1);

		float sumBatch = 0.0f;
		for (int i = 0; i < batches; i++) {
			Tensor p = prev.subTensor(i);
			Tensor r = real.subTensor(i);

			float ce = f(p, r, samples);
			sumBatch += -ce;
		}

		return new Tensor(1).set((sumBatch / batches), 0);
	}

	/**
	 * Cálculo interno da Cross Entropy.
	 * @param prev tensor com dados previstos.
	 * @param real tensor com dados reais.
	 * @param size quantidade de amostras.
	 * @return soma do CE.
	 */
	private float f(Tensor prev, Tensor real, int size) {
		double ce = 0.0;
		
		for (int i = 0; i < size; i++) {
			ce += real.get(i) * Math.log(prev.get(i) + eps);
		}

		return (float) ce;
	}
	
	@Override
	public Tensor backward(Tensor prev, Tensor real) {
		validateDimensions(prev, real);

		return prev.map(real,
			(p, r) -> (p - r)
		);
	}
}
