package rna.otimizadores;

import rna.estrutura.CamadaDensa;

/**
 * Classe base para implementações de otimizadores do treino da Rede Neural.
 * <p>
 *		O otimizador já deve levar em consideração que os gradientes foram 
 *		calculados previamente.
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
	 * @param parametros quantidade de pesos da rede neural.
	 */
	public void inicializar(CamadaDensa[] redec){
		throw new UnsupportedOperationException(
			"Inicialização do otimizador não implementada."
		);
	}

	/**
	 * Atualiza os pesos da Rede Neural de acordo com o otimizador configurado.
	 * <p>
	 *      A atualização de pesos é feita uma única vez em todos os parâmetros da rede.
	 * </p>
	 * Exemplo de uso:
	 * <pre>
	 *	public void atualizar(Camada[] redec){
	 * 	//indice na lista de coeficientes, caso o otimizador precise	
	 * 	int id = 0;
	 *		for(Camada camada : redec){
	 *			for(Neuronio neuronio : camada.neuronios()){
	 *				for(int i = 0; i < neuronio.pesos.length; i++){
	 *					// implementação do otimizador
	 *
	 *					id++;
	 *				}      
	 *			}
	 *		}
	 *	}
	 * </pre>
	 * @param redec Rede Neural em formato de lista de camadas.
	 */
	public void atualizar(CamadaDensa[] redec){
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
