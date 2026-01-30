package jnn.acts;

/**
 * Implementação da função de ativação GELU para uso dentro 
 * dos modelos.
 * <p>
 *    A função GELU (Gaussian Error Linear Unit) é uma função de ativação 
 *    aproximada que parece resolver o problema de desaparecimento do gradiente
 *    melhor do que a ReLU.
 * </p>
 */
public class GELU extends Ativacao {

	private final float RAIZ_2_POR_PI = (float) Math.sqrt(2 / Math.PI); 
	private final float ALFA = 0.044715f;

	/**
	 * Instancia uma nova função de ativação GELU.
	 */
	public GELU() { }

	@Override
	protected float fx(float x) {
		float xCubo = x * x * x;
		float tanh = tanh(RAIZ_2_POR_PI * (x + ALFA * xCubo));
		return 0.5f * x * (1.0f + tanh);
	}

	@Override
	protected float dx(float x) {
		float xCubo = x * x * x;
		float tanh = tanh(RAIZ_2_POR_PI * (x + ALFA * xCubo));
		float exp = (float) Math.exp(-0.5 * x * x) / RAIZ_2_POR_PI;
		return 0.5f * (1.0f + tanh + x * exp);
	}

	final private float tanh(float x) {
		return (2.0f / (1.0f + (float) Math.exp(-2.0 * x))) - 1.0f;
	}
}
