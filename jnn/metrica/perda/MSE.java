package jnn.metrica.perda;

import jnn.core.tensor.Tensor;

/**
 * Função de perda Mean Squared Error, calcula o erro médio 
 * quadrado entre as previsões e os valores reais.
 */
public class MSE extends Perda {

	/**
	 * Inicializa a função de perda Mean Squared Error.
	 */
	public MSE() {}

	@Override
	public Tensor forward(Tensor prev, Tensor real) {
		verificarDimensoes(prev, real);

		if (prev.numDim() == 1) {
			final int tam = prev.tam();
			double mse = f(prev, real, tam);
			
			return new Tensor(
				new double[]{ (mse/tam) }
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
	 * Calculo interno do mse.
	 * @param prev tensor com dados previstos.
	 * @param real tensor com dados reais.
	 * @param tam quantidade de amostras.
	 * @return soma do mse.
	 */
	private double f(Tensor prev, Tensor real, int tam) {
		double mse = 0.0;
		for (int i = 0; i < tam; i++) {
			double d = prev.get(i) - real.get(i);
			mse += d * d;
		}

		return mse;
	}
	
	@Override
	public Tensor backward(Tensor prev, Tensor real) {
		verificarDimensoes(prev, real);

		if (prev.numDim() == 1) {
			final int tam = prev.tam();
	
			return prev.map(
				real,
				(p, r) -> (2.0 / tam) * (p - r)
			);

		} else {
			final int lotes = prev.tamDim(0);
			final int amostras = prev.tamDim(1);

			return prev.map(real,
				(p, r) -> (2.0 / amostras) * (p - r) / lotes
			);
		}
	}
}
