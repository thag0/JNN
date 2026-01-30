package jnn.acts;

/**
 * Implementação da função de ativação SELU para uso dentro 
 * dos modelos.
 * <p>
 *    A função SELU (Scaled Exponential Linear Unit) é uma variação da ReLU
 *    com escala. Ela retorna o valor multiplicado por alpha (se for positivo)
 *    ou multiplicado por alpha vezes beta vezes a exponencial do valor menos
 *    1 (se for negativo).
 * </p>
 */
public class SELU extends Ativacao {

	/**
	 * Constante alfa;
	 */
	private float alfa;

	/**
	 * Fator de escala
	 */
	private float escala;

	/**
	 * Instancia a função de ativação SELU com os parâmetros alpha e scale especificados.
	 * @param alfa constante alfa.
	 * @param escala constante de escala.
	 */
	public SELU(Number alfa, Number escala) {
		this.alfa = alfa.floatValue();
		this.escala = escala.floatValue();
	}

	/**
	 * Instancia a função de ativação SELU.
	 */
	public SELU() {
		this(1.673263, 1.050700);
	}

	@Override
	protected float fx(float x) {
		return (x > 0.0) ? escala * x : escala * alfa * (float) (Math.exp(x) - 1.0);
	}

	@Override
	protected float dx(float x) {
		return (x > 0.0) ? escala : escala * alfa * (float) Math.exp(x);
	}

}
