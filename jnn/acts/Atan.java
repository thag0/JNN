package jnn.acts;

/**
 * Implementação da função de ativação Arco Tangente (atan) para uso dentro 
 * dos modelos.
 * <p>
 *    A função Arco Tangente retorna o ângulo cuja tangente é o número
 *    recebido como argumento. Ela retorna valores no intervalo [-pi/2, pi/2].
 * </p>
 */
public class Atan extends Ativacao {

	/**
	 * Instancia a função de ativação ArcTangente (atan).
	 */
	public Atan() { }

	@Override
	protected float fx(float x) {
		return (float) Math.atan(x); 
	}

	@Override
	protected float dx(float x) {
		return 1.0f / (1.0f + (x * x)); 
	}

}
