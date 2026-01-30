package jnn.acts;

/**
 * Implementação da função de ativação Swish para uso dentro dos modelos.
 * <p>
 *    A função Swish é uma função de ativação que gradualmente se aproxima 
 *    da função identidade à medida que seu argumento se torna positivo e 
 *    se assemelha à função sigmoid quando o argumento é negativo.
 * </p>
 */
public class Swish extends Ativacao {

	/**
	 * Instancia a função de ativação Swish.
	 */
	public Swish() { }

	@Override
	protected float fx(float x) {
		return x * sigmoid(x);
	}

	@Override
	protected float dx(float x) {
		float sig = sigmoid(x);
		return sig + (x * sig * (1.0f - sig));		
	}

	final private float sigmoid(float x) {
		return 1.0f / (float) (1.0 + Math.exp(-x));
	}
}
