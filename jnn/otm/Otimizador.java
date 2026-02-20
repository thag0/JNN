package jnn.otm;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

/**
 * <p>Otimizador base</p>
 * <p>
 *		Novos otimizadores devem implementar os métodos {@code construir()} 
 *		e {@code atualizar()} que são chamados obrigatoriamente no momento da 
 *		compilação e treino dos modelos.
 * </p>
 */
public abstract class Otimizador {

	/**
	 * Conjunto de elementos que serão otimizados.
	 */
	protected Tensor[] _params = {};

	/**
	 * Conjunto de gradientes dos parâmetros.
	 */
	protected Tensor[] _grads = {};

	/**
	 * Buffer de informações sobre o otimizador.
	 */
	protected StringBuilder info;

	/**
	 * Espaçamento para uma melhor formatação das informações do otimizador
	 */
	protected String pad = " ".repeat(4);

	/**
	 * Auxiliar para o controle de inicialização do otimizador.
	 */
	protected boolean _construido = false;

	/**
	 * Construtor privado.
	 */
	protected Otimizador() {}

	/**
	 * Verifica se o otimizador pode ser utilizado.
	 */
	protected void checkInicial() {
		if (!_construido) {
			throw new IllegalStateException(
				"\nOtimizador deve ser construído antes de ser utilizado, utilize construir()."
			);
		}
	}

	/**
	 * Captura os parâmetros e gradientes e inicializa os
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
			if (!params[i].compShape(grads[i])) {
				throw new IllegalArgumentException(
					"\nParâmetro " + i + params[i].shapeStr() + " deve conter" +
					" o mesmo formato do gradiente " + i + grads[i].shapeStr()
				);
			}
		}

		for (Tensor p : params) {
			_params = JNNutils.addEmArray(_params, p);	
		}

		for (Tensor g : grads) {
			_grads = JNNutils.addEmArray(_grads, g);
		}
	}

	/**
	 * Inicializa os parâmetros necessários do otimizador.
	 * @param params array de {@code Tensor} contendo os parâmetros desejados.
	 * @param grads array de {@code Tensor} contendo os gradientes relacionados aos parâmetros.
	 */
	public abstract void construir(Tensor[] params, Tensor[] grads);

	/**
	 * Executa um passo de atualização do otimizador.
	 */
	public abstract void update();

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
		
		info
		.append("Otimizador")
		.append(" = [\n")
		.append(pad)
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
	 * Exibe, {@code via terminal}, as informações do otimizador.
	 */
	public void print() {
		System.out.println(info() + "]");
	}

	/**
	 * Retorna o nome do otimizador.
	 * @return nome do otimizador.
	 */
	public String nome() {
		return getClass().getSimpleName();
	}
}
