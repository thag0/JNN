package jnn.avaliacao.perda;

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
		super.verificarDimensoes(prev, real);
		int tam = prev.tam();
		
		double mse = 0.0;
		for (int i = 0; i < tam; i++) {
			double d = prev.get(i) - real.get(i);
			mse += d * d;
		}
		
		return new Tensor(new double[]{ (mse/tam) }, 1);
	}
	
	@Override
	public Tensor backward(Tensor prev, Tensor real) {
		super.verificarDimensoes(prev, real);
		final int tam = prev.tam();

		return prev.map(
			real,
			(p, r) -> (2.0 / tam) * (p-r)
		);
	}
}
