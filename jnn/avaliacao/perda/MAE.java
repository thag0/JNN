package jnn.avaliacao.perda;

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
	public double calcular(double[] previsto, double[] real) {
		super.verificarDimensoes(previsto, real);
		int tam = previsto.length;
		
		double mae = 0;
		for (int i = 0; i < tam; i++) {
			mae += Math.abs(previsto[i] - real[i]);
		}
		
		return mae / tam;
	}
	
	@Override
	public double[] derivada(double[] previsto, double[] real) {
		super.verificarDimensoes(previsto, real);

		double[] derivadas = new double[previsto.length];
		for (int i = 0; i < previsto.length; i++) {
			derivadas[i] = previsto[i] - real[i];
		}

		return derivadas;
	}
}
