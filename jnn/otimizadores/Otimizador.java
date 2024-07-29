package jnn.otimizadores;

import jnn.core.Utils;
import jnn.core.tensor.Tensor;

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
	 * @param params array de {@code Tensor} contendo os parâmetros desejados.
	 * @param grads array de {@code Tensor} contendo os gradientes desejados.
	 */
	protected void initParams(Tensor[] params, Tensor[] grads) {
		if (params.length != grads.length) {
			throw new IllegalStateException(
				"\nQuantidade de parâmetros e gradientes deve ser igual. " + 
				"\nRecebido: p = " + params.length + ", g = " + grads.length
			);
		}

		int n = params.length;
		for (int i = 0; i < n; i++) {
			if (!params[i].compararShape(grads[i])) {
				throw new IllegalArgumentException(
					"\nParâmetro " + i + params[i].shapeStr() + " deve conter" +
					" o mesmo formato do gradiente " + i + grads[i].shapeStr()
				);
			}
		}


		for (Tensor param : params) {
			_params = utils.addEmArray(_params, param);	
		}

		for (Tensor grad : grads) {
			_grads = utils.addEmArray(_grads, grad);
		}
	}

	/**
	 * Inicializa os parâmetros necessários do otimizador para as camadas 
	 * do modelo especificado.
	 * @param params array de {@code Tensor} contendo os parâmetros desejados.
	 * @param grads array de {@code Tensor} contendo os gradientes desejados.
	 */
	public abstract void construir(Tensor[] params, Tensor[] grads);

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
