package jnn.metrics.loss;

import jnn.core.tensor.Tensor;

/**
 * Função de perda Binary Cross Entropy, que é usada em problemas de 
 * classificação binária. Ela é comumente aplicada quando há apenas duas 
 * classes possíveis.
 */
 public class BinaryCrossEntropy extends Loss {
	float eps = 1e-7f;//evitar log 0

	/**
	 * Inicializa a função de perda Binary Cross Entropy.
	 */
	public BinaryCrossEntropy() {}

	@Override
	public Tensor forward(Tensor prev, Tensor real) {
		validateDimensions(prev, real);

        if (prev.numDim() == 1) {
            int len = prev.len();
            float bce = f(prev, real, len);
            
			return new Tensor(1).set((-bce / len), 0);
        }

		int batches = prev.lenDim(0);
		int samples = prev.lenDim(1);

		float sumBatch = 0.0f;
		for (int i = 0; i < batches; i++) {
			Tensor p = prev.subTensor(i);
			Tensor r = real.subTensor(i);
			
			float sum = f(p, r, samples);
			sumBatch += -sum / samples;
		}


		return new Tensor(1).set((sumBatch / batches), 0);
	}

	/**
	 * 
	 * @param prev
	 * @param real
	 * @param tam
	 * @return
	 */
	private float f(Tensor prev, Tensor real, int tam) {
        double bce = 0.0f;
        
		for (int i = 0; i < tam; i++) {
            float p = prev.get(i);
            float r = real.get(i);
            bce += r * Math.log(p + eps) + (1.0 - r) * Math.log(1.0 - p + eps);
        }

        return (float) bce;
	}

	@Override
	public Tensor backward(Tensor prev, Tensor real) {
		validateDimensions(prev, real);
		
		if (prev.numDim() == 1) {
			final int tam = prev.len();
			return prev.map(real,
				(p, r) -> (((1.0f - r) / (1.0f - p)) - (r / p)) / tam
			);
		} 
		
		int sampes = prev.lenDim(1);
		return prev.map(real,
			(p, r) -> (((1.0f - r) / (1.0f - p + eps)) - (r / (p + eps))) / sampes
		);
	}
}
