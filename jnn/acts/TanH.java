package jnn.acts;

import jnn.camadas.Densa;
import jnn.core.tensor.Tensor;

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
	protected float fx(float x) {
		return 2.0f / (float) (1.0 + Math.exp(-2.0 * x)) - 1.0f;
	}

	@Override
	protected float dx(float x) {
		float t = fx(x);
		return 1.0f - (t * t);		
	}

	@Override
	public void backward(Densa densa, Tensor g) {
		//aproveitar os resultados pre calculados
		float[] t = densa._saida.array();
		float[] grad = g.array();
		float[] d = new float[t.length];

		float ti;
		for (int i = 0; i < d.length; i++) {
			ti = t[i];
			d[i] = (1.0f - (ti * ti)) * grad[i];
		}

		densa._gradSaida.copiarElementos(d);
	}
}
