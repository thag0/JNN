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
	 * Instancia a função de ativação SELU com os parâmetros alpha e scale especificados.
	 * @param alfa constante alfa.
	 * @param escala constante de escala.
	 */
	public SELU(Number alfa, Number escala) {
		double a = alfa.doubleValue();
		double e = escala.doubleValue();

		construir(
			x -> (x > 0.0) ? e * x : e * a * (Math.exp(x) - 1.0),
			x -> (x > 0.0) ? e     : e * a * Math.exp(x)
		);
	}

	/**
	 * Instancia a função de ativação SELU.
	 */
	public SELU() {
		this(1.67326324, 1.05070098);
	}

}
