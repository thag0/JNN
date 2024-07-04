package jnn.otimizadores;

import jnn.camadas.Camada;
import jnn.core.tensor.Variavel;

/**
 * <h2>
 *    Stochastic Gradient Descent 
 * </h2>
 * <p>
 *    Implementação do otimizador do gradiente estocástico com momentum e
 *    acelerador de nesterov.
 * </p>
 * <p>
 *    O SGD funciona usando a seguinte expressão:
 * </p>
 * <pre>
 *m += (m * M) - (g * tA)
 *v += m // apenas com momentum
 *v += (M * m) - (g * tA) // com nesterov
 * </pre>
 * Onde:
 * <p>
 *    {@code v} - variável que será otimizada.
 * </p>
 * <p>
 *    {@code M} - valor de taxa de momentum (ou constante de momentum) 
 *    do otimizador.
 * </p>
 * <p>
 *    {@code m} - valor de momentum da correspondente a variável que será
 *    otimizada.
 * </p>
 * <p>
 *    {@code g} - gradientes correspondente a variável que será otimizada.
 * </p>
 * <p>
 *    {@code tA} - taxa de aprendizado do otimizador.
 * </p>
 */
public class SGD extends Otimizador {

	/**
	 * Taxa de aprendizado padrão do otimizador.
	 */
	private static final double PADRAO_TA = 0.01;

	/**
	 * Taxa de momentum padrão do otimizador.
	 */
	private static final double PADRAO_MOMENTUM = 0.9;

	/**
	 * Uso do acelerador de nesterov padrão.
	 */
	private static final boolean PADRAO_NESTEROV = false;

	/**
	 * Valor de taxa de aprendizado do otimizador.
	 */
	private final double tA;

	/**
	 * Valor de taxa de momentum do otimizador.
	 */
	private final double momentum;

	/**
	 * Usar acelerador de Nesterov.
	 */
	private final boolean nesterov;

	/**
	 * Coeficientes de momentum para os kernels.
	 */
	private Variavel[] m;
	
	/**
	 * Coeficientes de momentum para os bias.
	 */
	private Variavel[] mb;

	/**
	 * Inicializa uma nova instância de otimizador <strong> Stochastic Gradient 
	 * Descent (SGD) </strong> usando os valores de hiperparâmetros fornecidos.
	 * @param tA taxa de aprendizado do otimizador.
	 * @param m taxa de momentum do otimizador.
	 * @param nesterov usar acelerador de nesterov.
	 */
	public SGD(double tA, double m, boolean nesterov) {
		if (tA <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de aprendizado (" + tA + ") inválida."
			);
		}
		
		if (m < 0) {         
			throw new IllegalArgumentException(
				"\nTaxa de momentum (" + m + ") inválida."
			);
		}

		this.tA = tA;
		this.momentum = m;
		this.nesterov = nesterov;
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> Stochastic Gradient 
	 * Descent (SGD) </strong> usando os valores de hiperparâmetros fornecidos.
	 * @param tA taxa de aprendizado do otimizador.
	 * @param m taxa de momentum do otimizador.
	 */
	public SGD(double tA, double m) {
		this(tA, m, PADRAO_NESTEROV);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> Stochastic Gradient 
	 * Descent (SGD) </strong> usando os valores de hiperparâmetros fornecidos.
	 * @param tA taxa de aprendizado do otimizador.
	 */
	public SGD(double tA) {
		this(tA, PADRAO_MOMENTUM, PADRAO_NESTEROV);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> Stochastic Gradient 
	 * Descent (SGD) </strong>.
	 * <p>
	 *    Os hiperparâmetros do SGD serão inicializados com seus os valores padrão.
	 * </p>
	 */
	public SGD() {
		this(PADRAO_TA, PADRAO_MOMENTUM, PADRAO_NESTEROV);
	}

	@Override
	public void construir(Camada[] camadas) {
		int[] params = initParams(camadas);
		int kernels = params[0];
		int bias = params[1];
		
		m  = initVars(kernels);
		mb = initVars(bias);
		
		_construido = true;// otimizador pode ser usado
	}

	@Override
	public void atualizar() {
		verificarConstrucao();

		int idKernel = 0, idBias = 0;
		
		for (Camada camada : _params) {
			Variavel[] kernel = camada.kernelParaArray();
			Variavel[] gradK = camada.gradKernelParaArray();
			idKernel = sgd(kernel, gradK, m, idKernel);

			if (camada.temBias()) {
				Variavel[] bias = camada.biasParaArray();
				Variavel[] gradB = camada.gradBiasParaArray();
				idBias = sgd(bias, gradB, mb, idBias);
			}
		}
	}

	/**
	 * Atualiza as variáveis usando o gradiente pré calculado.
	 * @param vars variáveis que serão atualizadas.
	 * @param grads gradientes das variáveis.
	 * @param m coeficientes de momentum.
	 * @param id índice inicial das variáveis dentro do array de momentums.
	 * @return índice final após as atualizações.
	 */
	private int sgd(Variavel[] vars, Variavel[] grads, Variavel[] m, int id) {
		double mid, g;

		for (int i = 0; i < vars.length; i++) {
			mid = m[id].get();
			g = grads[i].get();
		
			m[id].set((mid * momentum) - (g * tA));
			vars[i].add(nesterov ? (mid * momentum) - (g * tA) : mid);
		
			id++;
		}

		return id;
	}

	@Override
	public String info() {
		verificarConstrucao();
		construirInfo();
		
		addInfo("Lr: " + tA);
		addInfo("Momentum: " + momentum);
		addInfo("Nesterov: " + nesterov);

		return super.info();
	}

}
