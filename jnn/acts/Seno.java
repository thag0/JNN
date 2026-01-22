package jnn.acts;

/**
 * Implementação da função de ativação Seno para uso dentro 
 * dos modelos.
 * <p>
 *    A função Seno retorna o seno do valor recebido como entrada.
 * </p>
 */
public class Seno extends Ativacao {

	/**
	 * Instancia a função de ativação Seno.
	 */
	public Seno() { }

	@Override
	protected double fx(double x) {
		return Math.sin(x);
	}

	@Override
	public double dx(double x) {
		return Math.cos(x);
	}
}
