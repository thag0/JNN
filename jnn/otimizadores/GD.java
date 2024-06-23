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
 *    {@code tA} - taxa de aprendizagem do otimizador.
 * </p>
 */
public class GD extends Otimizador {

	/**
	 * Valor de taxa de aprendizagem do otimizador.
	 */
	private double taxaAprendizagem;

	/**
	 * Inicializa uma nova instância de otimizador da <strong> Descida do Gradiente </strong>
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA taxa de aprendizagem do otimizador.
	 */
	public GD(double tA) {
		if (tA <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de aprendizagem (" + tA + "), inválida."
			);
		}

		this.taxaAprendizagem = tA;
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
		this._construido = true;// otimizador pode ser usado
	}

	@Override
	public void atualizar() {
		verificarConstrucao();
		
		for (Camada camada : _camadas) {
			Variavel[] kernel = camada.kernelParaArray();
			Variavel[] gradK = camada.gradKernelParaArray();		
			for (int i = 0; i < kernel.length; i++) {
				kernel[i].sub(gradK[i].get() * taxaAprendizagem);
			}

			if (camada.temBias()) {
				Variavel[] bias = camada.biasParaArray();
				Variavel[] gradB = camada.gradBiasParaArray();
				for (int i = 0; i < bias.length; i++) {
					bias[i].sub(gradB[i].get() * taxaAprendizagem);
				}
			}
		} 
	}

	@Override
	public String info() {
		verificarConstrucao();
		construirInfo();
		
		addInfo("TaxaAprendizagem: " + taxaAprendizagem);

		return super.info();
	}
	
}
