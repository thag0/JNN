package jnn.ativacoes;

/**
 * Implementação da função de ativação ArcTangente (atan) para uso dentro 
 * dos modelos.
 * <p>
 *    A função ArcTangente (atan) retorna o ângulo cuja tangente é o número
 *    recebido como argumento. Ela retorna valores no intervalo [-pi/2, pi/2].
 * </p>
 */
public class Atan extends Ativacao {

	/**
	 * Instancia a função de ativação ArcTangente (atan).
	 */
	public Atan() {
		construir(
			x -> Math.atan(x),
			x -> 1.0 / (1.0 + (x * x))
		);
	}
}
