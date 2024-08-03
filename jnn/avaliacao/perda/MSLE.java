package jnn.avaliacao.perda;

import jnn.core.tensor.Tensor;

/**
 * Função de perda  Mean Squared Logarithmic Error, calcula o 
 * erro médio quadrado logarítmico entre as previsões e os 
 * valores reais.
 */
public class MSLE extends Perda {

	/**
	 * Inicializa a função de perda Mean Squared Logarithmic Error.
	 */
	public MSLE() {}

	@Override
	public Tensor calcular(Tensor prev, Tensor real) {
		super.verificarDimensoes(prev, real);
		int tam = prev.tam();
		
		double emql = 0;
		for (int i = 0; i < tam; i++) {
			double d = Math.log(1 +  prev.get(i)) - Math.log(1 + real.get(i));
			emql += d * d;
		}
		
		return new Tensor(new double[]{ (emql/tam) }, 1);
	}
	
	@Override
	public Tensor derivada(Tensor prev, Tensor real) {
		super.verificarDimensoes(prev, real);
		final int tam = prev.tam();

		return prev.map(
			real,
			(p, r) -> (2.0 / tam) * (Math.log(1 + p) - Math.log(1 + r))
		);
	}
}
