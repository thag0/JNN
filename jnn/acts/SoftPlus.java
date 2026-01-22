package jnn.acts;

/**
 * Implementação da função de ativação SoftPlus para uso dentro dos modelos.
 * <p>
 *    A função SoftPlus é uma função de ativação suave, que suaviza a transição
 *    entre valores positivos e negativos.
 * </p>
 */
public class SoftPlus extends Ativacao {

	/**
	 * Instancia a função de ativação SoftPlus.
	 */
	public SoftPlus() { }

	@Override
	protected double fx(double x) {
		return Math.log(1.0 + Math.exp(x));
	}

	@Override
	protected double dx(double x) {
		double exp = Math.exp(x);
		return (exp / (1.0 + exp));
	}
}
