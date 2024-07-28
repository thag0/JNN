package jnn.otimizadores;

import jnn.core.Utils;
import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;

/**
 * Classe base para implementações de otimizadores do treino da biblioteca.
 * <p>
 *		O otimizador já deve levar em consideração que os gradientes para todas
 *		as camadas foram calculados previamente.
 * </p>
 * <p>
 *		Novos otimizadores devem implementar (pelo menos) os métodos {@code construir()} 
 *		e {@code atualizar()} que são chamados obrigatoriamente no momento da compilação e 
 *		treino dos modelos.
 * </p>
 * <p>
 *		O método {@code inicializar()} é útil para aqueles otimizadores que possuem atributos 
 *		especiais, como o coeficiente de momentum por exemplo.
 * </p>
 */
public abstract class Otimizador {

	/**
	 * Conjunto de elementos que serão otimizados.
	 */
	Tensor[] _params = {};

	/**
	 * Conjunto de gradientes dos parâmetros.
	 */
	Tensor[] _grads = {};

	/**
	 * Utilitário.
	 */
	Utils utils = new Utils();

	/**
	 * Buffer de informações sobre o otimizador.
	 */
	StringBuilder info;

	/**
	 * Espaçamento para uma melhor formatação das informações do otimizador
	 */
	String pad = " ".repeat(4);

	/**
	 * Auxiliar para o controle de inicialização do otimizador.
	 */
	protected boolean _construido = false;

	/**
	 * Verifica se o otimizador pode ser utilizado.
	 */
	protected void verificarConstrucao() {
		if (!_construido) {
			throw new IllegalStateException(
				"\nOtimizador deve ser construído."
			);
		}
	}

	/**
	 * Captura os parâmetros e gradientes do modelo e inicializa os
	 * atributos necessários para o otimizador.
	 * @param modelo modelo para otimização.
	 */
	protected void initParams(Modelo modelo) {
		Tensor[] p = modelo.params();
		Tensor[] g = modelo.grads();

		if (p.length != g.length) {
			throw new IllegalStateException(
				"\nQuantidade de parâmetros e gradientes deve ser igual. " + 
				"\nRecebido: p = " + p.length + ", g = " + g.length
			);
		}

		for (Tensor param : p) {
			_params = utils.addEmArray(_params, param);	
		}

		for (Tensor grad : g) {
			_grads = utils.addEmArray(_grads, grad);
		}
	}

	/**
	 * Inicializa os parâmetros necessários do otimizador para as camadas 
	 * do modelo especificado.
	 * @param modelo modelo para otimização.
	 */
	public abstract void construir(Modelo modelo);

	/**
	 * Executa um passo de atualização do otimizador.
	 */
	public abstract void atualizar();

	/**
 	 * Exibe as opções de configurações do otimizador.
	 * @return buffer formatado.
	 */
	public String info() {
		return info.toString();
	}

	/**
	 * Inicializa o valor padrão para informações do otimizador, informando
	 * seu nome como informação inicial.
	 */
	protected void construirInfo() {
		info = new StringBuilder();
		
		info.append(pad)
		.append("Otimizador: ")
		.append(nome())
		.append("\n");
	}

	/**
	 * Adiciona uma nova linha de informação do otimizador.
	 * @param info informação adicional do otimizador
	 */
	protected void addInfo(String info) {
		this.info.append(pad)
		.append(info)
		.append("\n");
	}

	/**
	 * Retorna o nome do otimizador.
	 * @return nome do otimizador.
	 */
	public String nome() {
		return getClass().getSimpleName();
	}
}
