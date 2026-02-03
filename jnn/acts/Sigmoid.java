package jnn.acts;

import jnn.camadas.Densa;
import jnn.core.tensor.Tensor;

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
	protected float fx(float x) {
		return 1.0f / (float) (1.0 + Math.exp(-x));
	}

	@Override
	protected float dx(float x) {
		float s = fx(x);
		return s * (1.0f - s);		
	}

	@Override
	public void backward(Densa densa, Tensor g) {
		//aproveitar os resultados pre calculados
		float[] s = densa._saida.array();
		float[] grad = g.array();
		float[] d = new float[s.length];

		float si;
		for (int i = 0; i < d.length; i++) {
			si = s[i];
			d[i] = (si * (1.0f - si)) * grad[i];
		}

		densa._gradSaida.copiarElementos(d);
	}
}
