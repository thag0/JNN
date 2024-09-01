package jnn.avaliacao.perda;

import jnn.core.tensor.Tensor;

/**
 * Função de perda Root Mean Squared Error, calcula a raiz do 
 * erro médio quadrado entre as previsões e os valores reais.
 */
public class RMSE extends Perda {
 
	/**
	 * Inicializa a função de perda Root Mean Squared Error.
	 */
	public RMSE() {}

	@Override
	public Tensor forward(Tensor prev, Tensor real) {
		super.verificarDimensoes(prev, real);
		int tam = prev.tam();
		
		double rmse = 0.0;
		for (int i = 0; i < tam; i++) {
			double d = prev.get(i) - real.get(i);
			rmse += d * d;
		}
		rmse /= tam;
		
		return new Tensor(new double[]{ Math.sqrt(rmse) }, 1);
	}
	 
	@Override
	public Tensor backward(Tensor prev, Tensor real) {
		super.verificarDimensoes(prev, real);
		final int tam = prev.tam();
		double rrmse = Math.sqrt(forward(prev, real).item());

		return prev.map(
			real,
			(p, r) -> (p - r) / (rrmse * tam)
		);
	}
}
