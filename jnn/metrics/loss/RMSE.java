package jnn.metrics.loss;

import jnn.core.tensor.Tensor;

/**
 * Função de perda Root Mean Squared Error, calcula a raiz do 
 * erro médio quadrado entre as previsões e os valores reais.
 */
public class RMSE extends Loss {
 
	/**
	 * Inicializa a função de perda Root Mean Squared Error.
	 */
	public RMSE() {}

	@Override
	public Tensor forward(Tensor prev, Tensor real) {
		validateDimensions(prev, real);
		
		if (prev.numDim() == 1) {
			int len = prev.len();
			float rmse = f(prev, real, len);
			
			return new Tensor(1)
			.set((float) Math.sqrt(rmse / len), 0);
		}

        int batches = prev.lenDim(0);
        int samples = prev.lenDim(1);

        float sumBatch = 0;
        for (int i = 0; i < batches; i++) {
            Tensor p = prev.subTensor(i);
            Tensor r = real.subTensor(i);

            float sum = f(p, r, samples) / samples;
            sumBatch += Math.sqrt(sum);
        }

        return new Tensor(1)
		.set((sumBatch / batches), 0);
	}

	/**
	 * Calculo interno do rmse.
	 * @param prev tensor com dados previstos.
	 * @param real tensor com dados reais.
	 * @param tam quantidade de amostras.
	 * @return soma do rmse.
	 */
	private float f(Tensor prev, Tensor real, int tam) {
		float rmse = 0.0f;
		for (int i = 0; i < tam; i++) {
			float d = prev.get(i) - real.get(i);
			rmse += d * d;
		}

		return rmse;
	}
	 
	@Override
	public Tensor backward(Tensor prev, Tensor real) {
		super.validateDimensions(prev, real);
		final int tam = prev.len();
		float rrmse = (float) Math.sqrt(forward(prev, real).item());

		return prev.map(
			real,
			(p, r) -> (p - r) / (rrmse * tam)
		);
	}
}
