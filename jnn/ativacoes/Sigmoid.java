package jnn.ativacoes;

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
	public Sigmoid() {
		construir(
			x -> 1.0 / (1.0 + Math.exp(-x)),
			x -> { 
				double s = 1.0 / (1.0 + Math.exp(-x));
				return s * (1.0 - s);
			}
		);
	}

	@Override
	public void backward(Densa densa) {
		//aproveitar os resultados pre calculados
		double[] s = densa._saida.paraArrayDouble();
		double[] g = densa._gradSaida.paraArrayDouble();
		double[] d = new double[s.length];

		double ei;
		for (int i = 0; i < d.length; i++) {
			ei = s[i];
			d[i] = (ei*(1 - ei)) * g[i];
		}

		densa._gradSaida.copiarElementos(d);
	}
}
