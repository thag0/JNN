package jnn.acts;

/**
 * Implementação da função de ativação LeakyReLU para uso dentro 
 * dos modelos.
 * <p>
 *    É possível configurar o valor de {@code alfa} para obter
 *    melhores resultados.
 * </p>
 */
public class LeakyReLU extends Ativacao {

	/**
	 * Constante alfa;
	 */
	private float alfa = 0.01f;

	/**
	 * Instancia a função de ativação LeakyReLU com seu valor de alfa configurável.
	 * <p>
	 *    A ativação LeakyReLU funciona semelhante a função ReLU, retornando o próprio 
	 *    valor recebido caso ele seja maior que um, mas caso contrário ela retorna um 
	 *    pequeno valor alfa que será multiplicado pela saída.
	 * </p>
	 * @param alfa novo valor alfa.
	 */
	public LeakyReLU(Number alfa) {
		this.alfa = alfa.floatValue();
	}

	/**
	 * Instancia a função de ativação LeakyReLU com o valor de alfa padrão.
	 * <p>
	 *    A ativação LeakyReLU funciona semelhante a função ReLU, retornando o próprio 
	 *    valor recebido caso ele seja maior que um, mas caso contrário ela retorna um 
	 *    pequeno valor alfa que será multiplicado pela saída.
	 * </p>
	 * <p>
	 *    O valor padrão para o alfa é {@code 0.01}.
	 * </p>
	 */
	public LeakyReLU() {
		this(0.01);
	}

	@Override
	protected float fx(float x) {
		return (x > 0.0) ? x : alfa * x;
	}

	@Override
	protected float dx(float x) {
		return (x > 0.0) ? 1.0f : alfa;
	}
}
