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
			x -> (2 / (1 + Math.exp(-2*x))) - 1, 
			x -> {
				double t = (2 / (1 + Math.exp(-2*x))) - 1;
				return 1 - (t * t);
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
			d[i] = (1 - (ti*ti)) * g[i];
		}

		densa._gradSaida.copiarElementos(d);
	}
}
