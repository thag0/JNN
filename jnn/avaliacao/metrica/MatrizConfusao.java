package jnn.avaliacao.metrica;

import jnn.core.tensor.Tensor;

/**
 *	Matriz de confusão para avaliação de modelos de classificação.
 */
public class MatrizConfusao extends Metrica {

	/**
	 * Instancia a métrica da <strong>Matriz de confusão</strong>.
	 * <p>
	 * 		A matriz de confusão é uma tabela que descreve o desempenho 
	 * 		de um modelo de classificação em relação a um conjunto de 
	 * 		dados de teste, onde cada linha da matriz representa as instâncias
	 * 		reais de cada classe e cada coluna representa as instâncias preditas.
	 * </p>
	 */
	public MatrizConfusao() {}

	@Override
	public Tensor forward(Tensor[] prev, Tensor[] real) {
		return super.matrizConfusao(prev, real);
	}

}
