package jnn.metrics.loss;

import jnn.core.tensor.Tensor;

/**
 * Função de perda Mean Absolute Error, calcula o erro médio 
 * absoluto entre as previsões e os valores reais.
 */
public class MAE extends Loss {

	/**
	 * Inicializa a função de perda Mean Absolute Error.
	 */
	public MAE() {}

	@Override
	public Tensor forward(Tensor prev, Tensor real) {
		validateDimensions(prev, real);

		if (prev.numDim() == 1) {
			final int tam = prev.len();
			float mae = f(prev, real, tam);
			return new Tensor(1).set((mae/tam), 0);
		
		} else {
			int batches = prev.lenDim(0);
			int samples = prev.lenDim(1);

			float sumBatch = 0;
			for (int i = 0; i < batches; i++) {
				Tensor p = prev.subTensor(i);
				Tensor r = real.subTensor(i);
				float sumSmaples = f(p, r, samples);

				sumBatch += sumSmaples / samples;
			}

			return new Tensor(1).set((sumBatch / batches), 0);
		}
	}

	/**
	 * Calculo interno do mae.
	 * @param prev tensor com dados previstos.
	 * @param real tensor com dados reais.
	 * @param tam quantidade de amostras.
	 * @return soma do mae.
	 */
	private float f(Tensor prev, Tensor real, int tam) {
		float mae = 0.0f;
		for (int i = 0; i < tam; i++) {
			mae += Math.abs(prev.get(i) - real.get(i));
		}

		return mae;
	}
	
	@Override
	public Tensor backward(Tensor prev, Tensor real) {
		validateDimensions(prev, real);

		if (prev.numDim() == 1) {
			final int tam = prev.lenDim(0);
			return prev.map(real,
				(p, r) -> {
					float d = p - r;
					if (d > 0) return  1 / tam;
					if (d < 0) return -1 / tam;
					return 0;
				}
			);
		
		} else {
			int batches = prev.lenDim(0);
			int samples = prev.lenDim(1);
			float scale = 1.0f / (samples * batches);

			return prev.map(real, (p, r) -> {
				float d = p - r;
				if (d > 0) return  scale;
				if (d < 0) return -scale;
				return 0.0f;
			});
		}
	}
}
