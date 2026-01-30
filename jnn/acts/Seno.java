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
	protected float fx(float x) {
		return (float) Math.sin(x);
	}

	@Override
	public float dx(float x) {
		return (float) Math.cos(x);
	}
}
