package jnn.metrica.perda;

import jnn.core.tensor.Tensor;

/**
 * <h2>
 *    Base para implementações de funções de perda
 * </h2>
 * <p>
 *    As funções de perda são usadas para avaliações de modelos
 *    e principalmente para o treinamento, calculando os gradientes
 *    necessários para a atualização de parâmetros dos modelos.
 * </p>
 */
public abstract class Perda {

	/**
	 * Calcula a função de perda configurada.
	 * @param prev {@code Tensor} com dados previstos.
	 * @param real {@code Tensor} com dados reais.
	 * @return {@code Tensor} contendo o valor de perda.
	 */
	public abstract Tensor forward(Tensor prev, Tensor real);

	/**
	 * Calcula a derivada da função de perda configurada.
	 * @param prev {@code Tensor} com dados previstos.
	 * @param real {@code Tensor} com dados reais.
	 * @return {@code Tensor} contendo os valores de derivada.
	 */
	public abstract Tensor backward(Tensor prev, Tensor real);

	/**
	 * Auxiliar para verificar se os tamanhos dos tensores que serão usados
	 * pelas funções de perda são iguais.
	 * @param prev {@code Tensor} com dados previstos.
	 * @param real {@code Tensor} com dados reais.
	 */
	protected void verificarDimensoes(Tensor prev, Tensor real) {
		final int dimsPrev = prev.numDim();
		final int dimsReal = real.numDim();

		if (dimsPrev > 2 || dimsReal > 2) {
			throw new UnsupportedOperationException(
				"\nOperação suportada para tensores 1D e 2D, mas " +
				" prev = " + dimsPrev + "D e real = " + dimsReal + "D."
			);
		}

		if (!prev.compShape(real)) {
			throw new IllegalArgumentException(
				"\nAs dimensões dos tensores não coincidem, prev = " + prev.shapeStr() +
				" e real = " + real.shapeStr()
			);
		}
	}

	/**
	 * Retorna o nome da função de perda.
	 * @return nome da função de perda.
	 */
	public String nome() {
		return getClass().getSimpleName();
	}
}
