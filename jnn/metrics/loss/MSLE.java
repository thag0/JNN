package jnn.metrics.loss;

import jnn.core.tensor.Tensor;

/**
 * Função de perda  Mean Squared Logarithmic Error, calcula o 
 * erro médio quadrado logarítmico entre as previsões e os 
 * valores reais.
 */
public class MSLE extends Loss {
	
	/**
	 * Pequeno valor para evitar log de zero.
	 */
	float eps = 1e-7f;

	/**
	 * Inicializa a função de perda Mean Squared Logarithmic Error.
	 */
	public MSLE() {}

	@Override
	public Tensor forward(Tensor prev, Tensor real) {
		validateDimensions(prev, real);
		
		if (prev.numDim() == 1) {
			int len = prev.len();
			float emql = f(prev, real, len);
		
			return new Tensor(1).set((emql/len), 0);
		}

        int batches = prev.lenDim(0);
        int samples = prev.lenDim(1);

        float sumBatch = 0;
        for (int i = 0; i < batches; i++) {
            Tensor p = prev.subTensor(i);
            Tensor r = real.subTensor(i);

            sumBatch += f(p, r, samples) / samples;
        }


        return new Tensor(1).set((sumBatch / batches), 0);
	}

	/**
	 * Calculo interno do msle.
	 * @param prev tensor com dados previstos.
	 * @param real tensor com dados reais.
	 * @param tam quantidade de amostras.
	 * @return soma do msle.
	 */
	private float f(Tensor prev, Tensor real, int tam) {
		double emql = 0;
		for (int i = 0; i < tam; i++) {
			double d = Math.log(1.0 +  prev.get(i)) - Math.log(1.0 + real.get(i));
			emql += d * d;
		}
		
		return (float) emql;
	}
	
	@Override
	public Tensor backward(Tensor prev, Tensor real) {
		validateDimensions(prev, real);
        
		final int len = prev.numDim() == 1 ? prev.lenDim(0) : prev.lenDim(1);

        return prev.map(real, (p, r) -> {
            float lp = (float) Math.log(1.0 + p + eps);
            float lr = (float) Math.log(1.0 + r + eps);
            return (2.0f / len) * (lp - lr) * (1.0f / (1.0f + p + eps));
        });
	}
}
