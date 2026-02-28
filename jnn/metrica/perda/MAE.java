package jnn.metrica.perda;

import jnn.core.tensor.Tensor;

/**
 * Função de perda Mean Absolute Error, calcula o erro médio 
 * absoluto entre as previsões e os valores reais.
 */
public class MAE extends Perda {

	/**
	 * Inicializa a função de perda Mean Absolute Error.
	 */
	public MAE() {}

	@Override
	public Tensor forward(Tensor prev, Tensor real) {
		verificarDimensoes(prev, real);

		if (prev.numDim() == 1) {
			final int tam = prev.tam();
			float mae = f(prev, real, tam);
			return new Tensor(1).set((mae/tam), 0);
		
		} else {
			final int lotes = prev.tamDim(0);
			final int amostras = prev.tamDim(1);

			float somaLote = 0;
			for (int i = 0; i < lotes; i++) {
				Tensor p = prev.subTensor(i);
				Tensor r = real.subTensor(i);
				float somaAmostras = f(p, r, amostras);

				somaLote += somaAmostras / amostras;
			}

			return new Tensor(1).set((somaLote / lotes), 0);
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
		verificarDimensoes(prev, real);

		if (prev.numDim() == 1) {
			final int tam = prev.tamDim(0);
			return prev.map(real,
				(p, r) -> {
					float d = p - r;
					if (d > 0) return  1 / tam;
					if (d < 0) return -1 / tam;
					return 0;
				}
			);
		
		} else {
			int lotes = prev.tamDim(0);
			int amostras = prev.tamDim(1);
			float escala = 1.0f / (amostras * lotes);

			return prev.map(real, (p, r) -> {
				float d = p - r;
				if (d > 0) return  escala;
				if (d < 0) return -escala;
				return 0.0f;
			});
		}
	}
}
