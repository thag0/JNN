package jnn.otimizadores;

import jnn.camadas.Camada;

/**
 * <h2>
 *    Adaptive Moment Estimation
 * </h2>
 * Implementação do algoritmo de otimização Adam.
 * <p>
 *    O algoritmo ajusta os pesos do modelo usando o gradiente descendente 
 *    com momento e a estimativa adaptativa de momentos de primeira e segunda ordem.
 * </p>
 * <p>
 * 	Os hiperparâmetros do Adam podem ser ajustados para controlar o 
 * 	comportamento do otimizador durante o treinamento.
 * </p>
 * <p>
 *    O Adam funciona usando a seguinte expressão:
 * </p>
 * <pre>
 *var -= (alfa * m) / ((√ v) + eps)
 * </pre>
 * Onde:
 * <p>
 *    {@code var} - variável que será otimizada.
 * </p>
 * <p>
 *    {@code alfa} - correção aplicada a taxa de aprendizagem.
 * </p>
 * <p>
 *    {@code m} - coeficiente de momentum correspondente a variável que
 *    será otimizada;
 * </p>
 * <p>
 *    {@code v} - coeficiente de momentum de segunda orgem correspondente 
 *    a variável que será otimizada;
 * </p>
 * <p>
 *    {@code eps} - pequeno valor usado para evitar divisão por zero.
 * </p>
 * O valor de {@code alfa} é dado por:
 * <pre>
 * alfa = taxaAprendizagem * √(1- beta1ⁱ) / (1 - beta2ⁱ)
 * </pre>
 * Onde:
 * <p>
 *    {@code i} - contador de interações do Adam.
 * </p>
 * As atualizações de momentum de primeira e segunda ordem se dão por:
 *<pre>
 *m += (1 - beta1) * (g  - m)
 *v += (1 - beta2) * (g² - v)
 *</pre>
 * Onde:
 * <p>
 *    {@code beta1 e beta2} - valores de decaimento dos momentums de primeira
 *    e segunda ordem.
 * </p>
 * <p>
 *    {@code g} - gradiente correspondente a variável que será otimizada.
 * </p>
 */
public class Adam extends Otimizador {

	/**
	 * Valor de taxa de aprendizagem padrão do otimizador.
	 */
	private static final double PADRAO_TA = 0.001;

	/**
	 * Valor padrão para o decaimento do momento de primeira ordem.
	 */
	private static final double PADRAO_BETA1 = 0.9;
 
	/**
	 * Valor padrão para o decaimento do momento de segunda ordem.
	 */
	private static final double PADRAO_BETA2 = 0.999;
	 
	/**
	 * Valor padrão para epsilon.
	 */
	private static final double PADRAO_EPS = 1e-7;

	/**
	 * Valor de taxa de aprendizagem do otimizador.
	 */
	private double taxaAprendizagem;

	/**
	 * Decaimento do momentum.
	 */
	private double beta1;
	 
	/**
	 * Decaimento do momentum de segunda ordem.
	 */
	private double beta2;
	 
	/**
	 * Usado para evitar divisão por zero.
	 */
	private double epsilon;

	/**
	 * Coeficientes de momentum para os kernels.
	 */
	private double[] m;

	/**
	 * Coeficientes de momentum para os bias.
	 */
	private double[] mb;

	/**
	 * Coeficientes de momentum de segunda ordem para os kernels.
	 */
	private double[] v;

	/**
	 * Coeficientes de momentum de segunda ordem para os bias.
	 */
	private double[] vb;
	
	/**
	 * Contador de iterações.
	 */
	long interacoes = 0;
 
	/**
	 * Inicializa uma nova instância de otimizador <strong> Adam </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA taxa de aprendizagem do otimizador.
	 * @param beta1 decaimento do momento de primeira ordem.
	 * @param beta2 decaimento do momento de segunda ordem.
	 * @param eps usado para evitar a divisão por zero.
	 */
	public Adam(double tA, double beta1, double beta2, double eps) {
		if (tA <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de aprendizagem (" + tA + "), inválida."
			);
		}
		if (beta1 <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de decaimento de primeira ordem (" + beta1 + "), inválida."
			);
		}
		if (beta2 <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de decaimento de segunda ordem (" + beta2 + "), inválida."
			);
		}
		if (eps <= 0) {
			throw new IllegalArgumentException(
				"\nEpsilon (" + eps + "), inválido."
			);
		}
		
		this.taxaAprendizagem = tA;
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.epsilon = eps;
	}
 
	/**
	 * Inicializa uma nova instância de otimizador <strong> Adam </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA taxa de aprendizagem do otimizador.
	 * @param beta1 decaimento do momento de primeira ordem.
	 * @param beta2 decaimento do momento de segunda ordem.
	 */
	public Adam(double tA, double beta1, double beta2) {
		this(tA, beta1, beta2, PADRAO_EPS);
	}
 
	/**
	 * Inicializa uma nova instância de otimizador <strong> Adam </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA taxa de aprendizagem do otimizador.
	 */
	public Adam(double tA) {
		this(tA, PADRAO_BETA1, PADRAO_BETA2, PADRAO_EPS);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> Adam </strong>.
	 * <p>
	 *    Os hiperparâmetros do Adam serão inicializados com os valores 
	 *    padrão.
	 * </p>
	 */
	public Adam() {
		this(PADRAO_TA, PADRAO_BETA1, PADRAO_BETA2, PADRAO_EPS);
	}

	@Override
	public void construir(Camada[] camadas) {
		int nKernel = 0;
		int nBias = 0;
		
		for (Camada camada : camadas) {
			if (!camada.treinavel()) continue;

			nKernel += camada.kernel().tamanho();
			if (camada.temBias()) nBias += camada.bias().tamanho();
		}

		this.m  = new double[nKernel];
		this.v  = new double[nKernel];
		this.mb = new double[nBias];
		this.vb = new double[nBias];

		_construido = true;//otimizador pode ser usado
	}

	@Override
	public void atualizar(Camada[] camadas) {
		verificarConstrucao();
		
		interacoes++;
		double forcaB1 = Math.pow(beta1, interacoes);
		double forcaB2 = Math.pow(beta2, interacoes);
		double alfa = taxaAprendizagem * Math.sqrt(1 - forcaB2) / (1 - forcaB1);
		
		int idKernel = 0, idBias = 0;
		for (Camada camada : camadas) {
			if (!camada.treinavel()) continue;

			double[] kernel = camada.kernelParaArray();
			double[] gradK = camada.gradKernelParaArray();
			idKernel = calcular(kernel, gradK, m, v, alfa, idKernel);
			camada.setKernel(kernel);
			
			if (camada.temBias()) {
				double[] bias = camada.biasParaArray();
				double[] gradB = camada.gradBiasParaArray();
				idBias = calcular(bias, gradB, mb, vb, alfa, idBias);
				camada.setBias(bias);
			}     
		}
	}

	/**
	 * Atualiza as variáveis usando o gradiente pré calculado.
	 * @param vars variáveis que serão atualizadas.
	 * @param grads gradientes das variáveis.
	 * @param m coeficientes de momentum de primeira ordem das variáveis.
	 * @param v coeficientes de momentum de segunda ordem das variáveis.
	 * @param alfa pequena correção na taxa de aprendizagem.
	 * @param id índice inicial das variáveis dentro do array de momentums.
	 * @return índice final após as atualizações.
	 */
	private int calcular(double[] vars, double[] grads, double[] m, double[] v, double alfa, int id) {
		double g;

		for (int i = 0; i < vars.length; i++) {
			g = grads[i];
			m[id] += (1 - beta1) * (g    - m[id]);
			v[id] += (1 - beta2) + ((g*g - v[id]));  
			vars[i] -= (alfa * m[id]) / (Math.sqrt(v[id]) + epsilon);
		
			id++;
		}

		return id;
	}

	@Override
	public String info() {
		verificarConstrucao();
		construirInfo();
		
		addInfo("TaxaAprendizagem: " + taxaAprendizagem);
		addInfo("Beta1: " + beta1);
		addInfo("Beta2: " + beta2);
		addInfo("Epsilon: " + epsilon);

		return super.info();
	}

}
