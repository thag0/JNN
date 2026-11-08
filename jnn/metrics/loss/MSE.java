package jnn.metrics.loss;

import jnn.core.tensor.Tensor;

/**
 * Função de perda Mean Squared Error, calcula o erro médio 
 * quadrado entre as previsões e os valores reais.
 */
public class MSE extends Loss {

	/**
	 * Inicializa a função de perda Mean Squared Error.
	 */
	public MSE() {}

	@Override
	public Tensor forward(Tensor prev, Tensor real) {
		validateDimensions(prev, real);

		if (prev.numDim() == 1) {
			final int tam = prev.len();
			float mse = f(prev, real, tam);
			
			return new Tensor(1).set((mse / tam), 0);
		} 

		final int lotes = prev.lenDim(0);
		final int amostras = prev.lenDim(1);

		float sumBatch = 0;
		for (int i = 0; i < lotes; i++) {
			Tensor p = prev.subTensor(i);
			Tensor r = real.subTensor(i);
			float sumSamples = f(p, r, amostras);

			sumBatch += sumSamples / amostras;
		}

		return new Tensor(1).set((sumBatch / lotes), 0);
	}

	/**
	 * Calculo interno do mse.
	 * @param prev tensor com dados previstos.
	 * @param real tensor com dados reais.
	 * @param tam quantidade de amostras.
	 * @return soma do mse.
	 */
	private float f(Tensor prev, Tensor real, int tam) {
		float mse = 0.0f;
		for (int i = 0; i < tam; i++) {
			float d = prev.get(i) - real.get(i);
			mse += d * d;
		}

		return mse;
	}
	
	@Override
	public Tensor backward(Tensor prev, Tensor real) {
		validateDimensions(prev, real);

		if (prev.numDim() == 1) {
			final int tam = prev.len();
			return prev.map(
				real,
				(p, r) -> (2.0f / tam) * (p - r)
			);
		} 
	
		int batches = prev.lenDim(0);
		int samples = prev.lenDim(1);

		return prev.map(real,
			(p, r) -> (2.0f / samples) * (p - r) / batches
		);
	}
}
