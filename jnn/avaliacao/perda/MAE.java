package jnn.avaliacao.perda;

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
		super.verificarDimensoes(prev, real);
		int tam = prev.tam();
		
		double mae = 0;
		for (int i = 0; i < tam; i++) {
			mae += Math.abs(prev.get(i) - real.get(i));
		}
		
		return new Tensor(
			new double[]{ (mae/tam) }
		);
	}
	
	@Override
	public Tensor backward(Tensor prev, Tensor real) {
		super.verificarDimensoes(prev, real);

		return prev.map(
			real,
			(p, r) -> (p - r)
		);
	}
}
