package jnn.acts;

/**
 * Implementação da função de ativação ELU para uso dentro 
 * dos modelos.
 * <p>
 *    É possível configurar o valor de {@code alfa} para obter
 *    melhores resultados.
 * </p>
 */
public class ELU extends Ativacao {

	/**
	 * Constante alfa.
	 */
	private double alfa = 0.01;

	/**
	 * Instancia a função de ativação ELU com 
	 * seu valor de alfa configurável.
	 * @param alfa novo valor alfa.
	 */
	public ELU(Number alfa) {
		this.alfa = alfa.doubleValue();
	}

	/**
	 * Instancia a função de ativação ELU com 
	 * seu valor de alfa padrão.
	 * <p>
	 *    O valor padrão para o alfa é {@code 0.01}.
	 * </p>
	 */
	public ELU() {
		this(0.01);
	}

	@Override
	protected double fx(double x) {
		return (x > 0.0) ? x : alfa * (Math.exp(x) - 1.0); 
	}

	@Override
	protected double dx(double x) {
		return (x > 0.0) ? 1.0 : alfa * Math.exp(x); 
	}

}
