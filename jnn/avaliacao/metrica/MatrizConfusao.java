package jnn.avaliacao.metrica;

import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;

/**
 * Calcula a matriz de confusão para avaliação de modelos de classificação.
 * <p>
 * 		A matriz de confusão é uma tabela que descreve o desempenho de um modelo de 
 * 		classificação em relação a um conjunto de dados de teste, onde cada linha da 
 * 		matriz representa as instâncias reais de cada classe e cada coluna representa 
 * 		as instâncias preditas.
 * </p>
 */
public class MatrizConfusao extends Metrica {

	public MatrizConfusao() {}

	@Override
	public Tensor calcularMatriz(Modelo modelo, Tensor[] entrada, Tensor[] saidas) {
		return super.matrizConfusao(modelo, entrada, saidas);
	} 
}
