package rna.otimizadores;

import rna.camadas.Camada;

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
 *		O método {@code inicializar()} é útil para aqueles otimizadores que possuem atributos 
 *		especiais, como o coeficiente de momentum por exemplo.
 * </p>
 */
public abstract class Otimizador{

	/**
	 * Informações sobre o otimizador.
	 */
	String info;

	/**
	 * Espaçamento para uma melhor formatação das informações do otimizador
	 */
	String pad = " ".repeat(4);

	/**
	 * Auxiliar para o controle de inicialização do otimizador.
	 */
	protected boolean construido = false;

	/**
	 * Verifica se o otimizador pode ser utilizado.
	 */
	protected void verificarConstrucao(){
		if(this.construido == false){
			throw new IllegalStateException(
				"O otimizador deve ser construído para poder ser usado."
			);
		}
	}

	/**
	 * Inicializa os parâmetros do otimizador para as camadas do modelo especificado.
	 * @param camadas array de camadas do modelo.
	 */
	public abstract void construir(Camada[] camadas);

	/**
	 * Atualiza os parâmetros treináveis do modelo de acordo com o 
	 * otimizador especificado.
	 * <p>
	 *		A atualização de parâmetros é feita uma única vez em todas
	 *		as camadas do modelo.
	 * </p>
	 * @param camadas array de camadas do modelo.
	 */
	public abstract void atualizar(Camada[] camadas);

	/**
 	 * Exibe as opções de configurações do otimizador.
	 * @return buffer formatado.
	 */
	public String info(){
		return this.info;
	}

	/**
	 * Inicializa o valor padrão para informações do otimizador, informando
	 * seu nome como informação inicial.
	 */
	protected void construirInfo(){
		this.info = "";
		this.info += this.pad +  "Otimizador: " + this.nome() + "\n";
	}

	/**
	 * Adiciona uma nova linha de informação do otimizador.
	 * @param info informação adicional do otimizador
	 */
	protected void addInfo(String info){
		this.info += this.pad + info + "\n";
	}

	/**
	 * Retorna o nome do otimizador.
	 * @return nome do otimizador.
	 */
	public String nome(){
		return getClass().getSimpleName();
	}
}
