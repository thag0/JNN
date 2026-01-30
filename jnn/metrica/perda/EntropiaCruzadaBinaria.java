package jnn.metrica.perda;

import jnn.core.tensor.Tensor;

/**
 * Função de perda Binary Cross Entropy, que é usada em problemas de 
 * classificação binária. Ela é comumente aplicada quando há apenas duas 
 * classes possíveis.
 */
 public class EntropiaCruzadaBinaria extends Perda {
	float eps = 1e-7f;//evitar log 0

	/**
	 * Inicializa a função de perda Binary Cross Entropy.
	 */
	public EntropiaCruzadaBinaria() {}

	@Override
	public Tensor forward(Tensor prev, Tensor real) {
		verificarDimensoes(prev, real);

        if (prev.numDim() == 1) {
            int tam = prev.tam();
            float ecb = f(prev, real, tam);
            return new Tensor(
				new float[]{ -ecb / tam }
			);

        }

		final int lotes = prev.tamDim(0);
		final int amostras = prev.tamDim(1);

		float somaLote = 0.0f;
		for (int i = 0; i < lotes; i++) {
			Tensor p = prev.subTensor(i);
			Tensor r = real.subTensor(i);
			
			float soma = f(p, r, amostras);
			somaLote += -soma / amostras;
		}

		return new Tensor(
			new float[]{ somaLote / lotes }
		);        
	}

	/**
	 * 
	 * @param prev
	 * @param real
	 * @param tam
	 * @return
	 */
	private float f(Tensor prev, Tensor real, int tam) {
        double ecb = 0.0f;
        for (int i = 0; i < tam; i++) {
            float p = prev.get(i);
            float r = real.get(i);
            ecb += r * Math.log(p + eps) + (1.0 - r) * Math.log(1.0 - p + eps);
        }

        return (float) ecb;
	}

	@Override
	public Tensor backward(Tensor prev, Tensor real) {
		verificarDimensoes(prev, real);
		
		if (prev.numDim() == 1) {
			final int tam = prev.tam();
			return prev.map(real,
				(p, r) -> (((1.0f - r) / (1.0f - p)) - (r / p)) / tam
			);
		} 
		
		int amostras = prev.tamDim(1);
		return prev.map(real,
			(p, r) -> (((1.0f - r) / (1.0f - p + eps)) - (r / (p + eps))) / amostras
		);
	}
}
