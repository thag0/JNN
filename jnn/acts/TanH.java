package jnn.acts;

import jnn.camadas.Densa;

/**
 * Implementação da função de ativação Tangente Hiperbólica (TanH) 
 * para uso dentro dos modelos.
 * <p>
 *    A função TanH retorna um valor na faixa de -1 a 1, sendo uma 
 *    versão suavizada da função sigmoid.
 * </p>
 */
public class TanH extends Ativacao {

	/**
	 * Instancia a função de ativação TanH.
	 */
	public TanH() {	}

	@Override
	protected double fx(double x) {
		return 2.0 / (1.0 + Math.exp(-2.0 * x)) - 1.0;
	}

	@Override
	protected double dx(double x) {
		double t = fx(x);
		return 1.0 - (t * t);		
	}

	@Override
	public void backward(Densa densa) {
		//aproveitar os resultados pre calculados
		double[] t = densa._saida.array();
		double[] g = densa._gradSaida.array();
		double[] d = new double[t.length];

		double ti;
		for (int i = 0; i < d.length; i++) {
			ti = t[i];
			d[i] = (1.0 - (ti*ti)) * g[i];
		}

		densa._gradSaida.copiarElementos(d);
	}
}
