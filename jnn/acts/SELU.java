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
	private double alfa;

	/**
	 * Fator de escala
	 */
	private double escala;

	/**
	 * Instancia a função de ativação SELU com os parâmetros alpha e scale especificados.
	 * @param alfa constante alfa.
	 * @param escala constante de escala.
	 */
	public SELU(Number alfa, Number escala) {
		this.alfa = alfa.doubleValue();
		this.escala = escala.doubleValue();
	}

	/**
	 * Instancia a função de ativação SELU.
	 */
	public SELU() {
		this(1.67326324, 1.05070098);
	}

	@Override
	protected double fx(double x) {
		return (x > 0.0) ? escala * x : escala * alfa * (Math.exp(x) - 1.0);
	}

	@Override
	protected double dx(double x) {
		return (x > 0.0) ? escala : escala * alfa * Math.exp(x);
	}

}
