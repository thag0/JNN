package jnn.otimizadores;

import jnn.camadas.Camada;
import jnn.core.tensor.Variavel;

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
	Camada[] _camadas = new Camada[0];

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
		if(!_construido){
			throw new IllegalStateException(
				"\nO otimizador deve ser construído para poder ser usado."
			);
		}
	}

	/**
	 * Calcula o número de parâmetros das camadas fornecidas e
	 * adiciona camadas treináveis ao otimizador.
	 * @param camadas {@code Camadas} treináveis.
	 * @return array contendo os valores dos parâmetros no formato
	 * {@code [kernels, bias]}.
	 */
	protected int[] initParams(Camada[] camadas) {
		int nKernel = 0;
		int nBias = 0;
		
		for (Camada camada : camadas) {
			if (!camada.treinavel()) continue;

			nKernel += camada.kernel().tamanho();
			if (camada.temBias()) nBias += camada.bias().tamanho();

			addCamada(camada);
		}

		return new int[]{
			nKernel, 
			nBias
		};
	}

	/**
	 * Inicializa um array de variáveis.
	 * @param tam tamanho desejado.
	 * @return array de acordo com o tamanho, zerado.
	 */
	protected Variavel[] initVars(int tam) {
		Variavel[] arr = new Variavel[tam];
		for (int i = 0; i < tam; i++) {
			arr[i] = new Variavel(0.0d);
		}

		return arr;
	}

	/**
	 * Inicializa os parâmetros do otimizador para as camadas do modelo especificado.
	 * @param camadas array de camadas do modelo.
	 */
	public abstract void construir(Camada[] camadas);

	/**
	 * Adiciona uma camada que será treinada aos parâmetros do otmizador.
	 * @param camada {@code Camada} treinável.
	 */
	protected void addCamada(Camada camada) {
		if (camada == null) {
			throw new IllegalArgumentException(
				"\nCamada não pode ser nula."
			);
		}

		Camada[] antigas = _camadas;
		_camadas = new Camada[antigas.length + 1];

		System.arraycopy(antigas, 0, _camadas, 0, antigas.length);
		_camadas[_camadas.length-1] = camada;
	}

	/**
	 * Atualiza os parâmetros inicializados do otimizador.
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
