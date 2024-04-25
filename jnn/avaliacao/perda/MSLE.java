package jnn.avaliacao.perda;

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
	public double calcular(double[] previsto, double[] real) {
		super.verificarDimensoes(previsto, real);
		int tam = previsto.length;
		
		double emql = 0;
		for (int i = 0; i < tam; i++) {
			double d = Math.log(1 + previsto[i]) - Math.log(1 + real[i]);
			emql += d * d;
		}
		
		return emql/tam;
	}
	
	@Override
	public double[] derivada(double[] previsto, double[] real) {
		super.verificarDimensoes(previsto, real);

		int tam = previsto.length;
		double[] derivadas = new double[previsto.length];
		for (int i = 0; i < tam; i++) {
			derivadas[i] = (2.0 / tam) * (Math.log(1 + previsto[i]) - Math.log(1 + real[i]));
		}
		return derivadas;
	}
}
