package jnn.otimizadores;

import jnn.camadas.Camada;
import jnn.core.tensor.Variavel;

/**
 * <h2>
 *    Gradient Descent
 * </h2>
 * Classe que implementa o algoritmo de Descida do Gradiente para otimização de redes neurais.
 * Atualiza diretamente os pesos da rede com base no gradiente.
 * <p>
 *    O Gradiente descendente funciona usando a seguinte expressão:
 * </p>
 * <pre>
 *    v -= g * tA
 * </pre>
 * Onde:
 * <p>
 *    {@code v} - variável que será otimizadada.
 * </p>
 *    {@code g} - gradiente correspondente a variável que será otimizada.
 * </p>
 * <p>
 *    {@code tA} - taxa de aprendizado do otimizador.
 * </p>
 */
public class GD extends Otimizador {

	/**
	 * Valor de taxa de aprendizado do otimizador.
	 */
	private final double tA;

	/**
	 * Inicializa uma nova instância de otimizador da <strong> Descida do Gradiente </strong>
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA taxa de aprendizado do otimizador.
	 */
	public GD(double tA) {
		if (tA <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de aprendizado (" + tA + ") inválida."
			);
		}

		this.tA = tA;
	}

	/**
	 * Inicializa uma nova instância de otimizador da <strong> Descida do Gradiente </strong>.
	 * <p>
	 *    Os hiperparâmetros do GD serão inicializados com os valores padrão.
	 * </p>
	 */
	public GD() {
		this(0.1);
	}

	@Override
	public void construir(Camada[] camadas) {
		initParams(camadas);
		_construido = true;// otimizador pode ser usado
	}

	@Override
	public void atualizar() {
		verificarConstrucao();
		
		for (Camada camada : _params) {
			Variavel[] kernel = camada.kernelParaArray();
			Variavel[] gradK = camada.gradKernelParaArray();		
			gd(kernel, gradK);

			if (camada.temBias()) {
				Variavel[] bias = camada.biasParaArray();
				Variavel[] gradB = camada.gradBiasParaArray();
				gd(bias, gradB);
			}
		} 
	}

    /**
	 * Atualiza as variáveis usando o gradiente pré calculado.
	 * @param vars variáveis que serão atualizadas.
	 * @param grads gradientes das variáveis.
	 */
	private void gd(Variavel[] vars, Variavel[] grads) {
		for (int i = 0; i < vars.length; i++) {
			vars[i].sub(grads[i].get() * tA);
		}
	}

	@Override
	public String info() {
		verificarConstrucao();
		construirInfo();
		
		addInfo("Lr: " + tA);

		return super.info();
	}
	
}
