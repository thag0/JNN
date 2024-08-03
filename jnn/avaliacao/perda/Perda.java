package jnn.avaliacao.perda;

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
	public abstract Tensor calcular(Tensor prev, Tensor real);

	/**
	 * Calcula a derivada da função de perda configurada.
	 * @param prev {@code Tensor} com dados previstos.
	 * @param real {@code Tensor} com dados reais.
	 * @return {@code Tensor} contendo os valores de derivada.
	 */
	public abstract Tensor derivada(Tensor prev, Tensor real);

	/**
	 * Auxiliar para verificar se os tamanhos dos tensores que serão usados
	 * pelas funções de perda são iguais.
	 * @param prev {@code Tensor} com dados previstos.
	 * @param real {@code Tensor} com dados reais.
	 */
	protected void verificarDimensoes(Tensor prev, Tensor real) {
		if (prev.numDim() != 1 || real.numDim() != 1) {
			throw new UnsupportedOperationException(
				"\nAmbos os tensores devem ter apenas uma dimensão."
			);
		}

		if (prev.tam() != real.tam()) {
			throw new IllegalArgumentException(
				"Dimensões de dados previstos (" + prev.tam() + 
				") diferente da dimensão dos dados reais (" + real.tam() + 
				")"
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
