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
			double mae = f(prev, real, tam);
			
			return new Tensor(
				new double[]{ (mae/tam) }
			);
		
		} else {
			final int lotes = prev.tamDim(0);
			final int amostras = prev.tamDim(1);

			double somaLote = 0;
			for (int i = 0; i < lotes; i++) {
				Tensor p = prev.subTensor(i);
				Tensor r = real.subTensor(i);
				double somaAmostras = f(p, r, amostras);

				somaLote += somaAmostras / amostras;
			}

			return new Tensor(
				new double[]{somaLote / lotes}
			);
		}
	}

	/**
	 * Calculo interno do mae.
	 * @param prev tensor com dados previstos.
	 * @param real tensor com dados reais.
	 * @param tam quantidade de amostras.
	 * @return soma do mae.
	 */
	private double f(Tensor prev, Tensor real, int tam) {
		double mae = 0.0;
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
					double d = p - r;
					if (d > 0) return 1 / tam;
					if (d < 0) return -1 / tam;
					return 0;
				}
			);
		
		} else {
			int lotes = prev.tamDim(0);
			int amostras = prev.tamDim(1);
			double escala = 1.0 / (amostras * lotes);

			return prev.map(real, (p, r) -> {
				double d = p - r;
				if (d > 0) return  escala;
				if (d < 0) return -escala;
				return 0.0;
			});
		}
	}
}
