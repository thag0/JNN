package jnn.ativacoes;

/**
 * Implementação da função de ativação GELU para uso dentro 
 * dos modelos.
 * <p>
 *    A função GELU (Gaussian Error Linear Unit) é uma função de ativação 
 *    aproximada que parece resolver o problema de desaparecimento do gradiente
 *    melhor do que a ReLU.
 * </p>
 */
public class GELU extends Ativacao {

	private final double RAIZ_2_POR_PI = Math.sqrt(2 / Math.PI); 
	private final double ALFA = 0.044715;

	/**
	 * Instancia uma nova função de ativação GELU.
	 */
	public GELU() {
		construir(this::gelu, this::gelud);
	}

	private double gelu(double x) {
		double xCubo = x * x * x;
		double tanh = tanh(RAIZ_2_POR_PI * (x + ALFA * xCubo));
		return 0.5 * x * (1.0 + tanh);
	}

	private double gelud(double x) {
		double xCubo = x * x * x;
		double tanh = tanh(RAIZ_2_POR_PI * (x + ALFA * xCubo));
		double exp = Math.exp(-0.5 * x * x) / RAIZ_2_POR_PI;
		return 0.5 * (1.0 + tanh + x * exp);
	}

	public double tanh(double x) {
		return (2 / (1 + Math.exp(-2*x))) - 1;
	}
}
