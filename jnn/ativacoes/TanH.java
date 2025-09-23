package jnn.ativacoes;

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
	public TanH() {
		construir(
			x -> (2.0 / (1.0 + Math.exp(-2.0 * x))) - 1.0, 
			x -> {
				double t = (2.0 / (1.0 + Math.exp(-2.0 * x))) - 1.0;
				return 1.0 - (t * t);
			}
		);
	}

	@Override
	public void backward(Densa densa) {
		//aproveitar os resultados pre calculados
		double[] t = densa._saida.paraArrayDouble();
		double[] g = densa._gradSaida.paraArrayDouble();
		double[] d = new double[t.length];

		double ti;
		for (int i = 0; i < d.length; i++) {
			ti = t[i];
			d[i] = (1.0 - (ti*ti)) * g[i];
		}

		densa._gradSaida.copiarElementos(d);
	}
}
