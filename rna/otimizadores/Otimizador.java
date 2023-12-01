package rna.otimizadores;

import rna.estrutura.Densa;

/**
 * Classe base para implementações de otimizadores do treino da Rede Neural.
 * <p>
 *		O otimizador já deve levar em consideração que os gradientes para todas
 *		as camadas foram calculados previamente.
 * </p>
 * <p>
 *		Novos otimizadores devem implementar (pelo menos) os métodos {@code inicialziar()} 
 *		e {@code atualizar()} que são chamados obrigatoriamente no momento da compilação e 
 *		treino da Rede Neural.
 * </p>
 * <p>
 *		O método {@code inicialziar()} é útil para aqueles otimizadores que possuem atributos 
 *		especiais, como o coeficiente de momentum por exemplo.
 * </p>
 */
public abstract class Otimizador{

	/**
	 * Inicializa os parâmetros do otimizador para que possa ser usado.
	 * @param redec Lista de camadas densas da Rede Neural.
	 */
	public void inicializar(Densa[] redec){
		throw new UnsupportedOperationException(
			"Inicialização do otimizador não implementada."
		);
	}

	/**
	 * Atualiza os pesos da Rede Neural de acordo com o otimizador configurado.
	 * <p>
	 *		A atualização de pesos é feita uma única vez em todos os parâmetros da rede.
	 * </p>
	 * @param redec Lista de camadas densas da Rede Neural.
	 */
	public void atualizar(Densa[] redec){
		throw new UnsupportedOperationException(
			"Método de atualização do otimizador não foi implementado."
		);
	}

	/**
 	 * Mostra as opções de configurações do otimizador.
	 * @return buffer formatado.
	 */
	public String info(){
		throw new UnsupportedOperationException(
			"Método de informações do otimizador não foi implementado."
		);
	}
}
