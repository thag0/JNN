package jnn.acts;

/**
 * Implementação da função de ativação Linear para uso dentro 
 * dos modelos.
 * <p>
 *    A função Linear apenas retorna o próprio valor de entrada
 *    para sua saída e não é indicada pra uso. Ela é boa para 
 *    fazer testes nas camadas que se seus resultados não alteram
 *    os valores recebidos.
 * </p>
 */
public class Linear extends Ativacao {

	/**
	 * Instancia a função de ativação Linear.
	 */
	public Linear() {
		construir(
			x -> x,
			_ -> 1.0
		);
	}
}
