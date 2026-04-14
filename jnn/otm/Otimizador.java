package jnn.otm;

import jnn.core.JNNutils;
import jnn.core.Parametro;
import jnn.core.tensor.Tensor;

/**
 * <p>
 *		Otimizador base
 * </p>
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
	protected Parametro[] _params = {};

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
	 * @param params array de {@code Parametro} para otimização.
	 */
	protected void initParams(Parametro[] params) {
		JNNutils.validarNaoNulo(params, "params == null.");
		
		for (int i = 0, n = params.length; i < n; i++) {
			Parametro p = params[i];
			Tensor w = p.weight;
			Tensor g = p.grad;

			if (!w.compShape(g)) {
				throw new IllegalArgumentException(
					"\nPeso " + i + w.shapeStr() + " deve conter" +
					" o mesmo formato do gradiente " + i + g.shapeStr()
				);
			}
		}

		_params = params;
	}

	/**
	 * Inicializa os parâmetros necessários do otimizador.
	 * @param params array de {@code Tensor} contendo os parâmetros desejados.
	 */
	public abstract void construir(Parametro[] params);

	/**
	 * Executa um passo de atualização do otimizador.
	 */
	public abstract void update();

	/**
	 * Retorna o valor atual da taxa de aprendizado (learning rate).
	 * @return valor do learning rate.
	 */
	public abstract float getLr();

	/**
	 * Configura um novo valor para a taxa de aprendizado (learning rate).
	 * @param lr novo learning rate.
	 */
	public abstract void setLr(float lr);

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
