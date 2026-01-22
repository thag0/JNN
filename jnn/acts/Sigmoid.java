package jnn.acts;

import jnn.camadas.Densa;

/**
 * Implementação da função de ativação Sigmóide para uso dentro 
 * dos modelos.
 * <p>
 *    A função Sigmóide é uma função de ativação bastante utilizada 
 *    em redes neurais, mapeando qualquer valor para o intervalo [0, 1].
 * </p>
 */
public class Sigmoid extends Ativacao {

	/**
	 * Instancia a função de ativação Sigmoid.
	 */
	public Sigmoid() { }

	@Override
	protected double fx(double x) {
		return 1.0 / (1.0 + Math.exp(-x));
	}

	@Override
	protected double dx(double x) {
		double s = fx(x);
		return s * (1.0 - s);		
	}

	@Override
	public void backward(Densa densa) {
		//aproveitar os resultados pre calculados
		double[] s = densa._saida.array();
		double[] g = densa._gradSaida.array();
		double[] d = new double[s.length];

		double si;
		for (int i = 0; i < d.length; i++) {
			si = s[i];
			d[i] = (si*(1.0 - si)) * g[i];
		}

		densa._gradSaida.copiarElementos(d);
	}
}
