package jnn.otimizadores;

import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;

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
 *    {@code alfa} - correção aplicada a taxa de aprendizado.
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
 * alfa = tA * √(1- beta1ⁱ) / (1 - beta2ⁱ)
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
	 * Valor de taxa de aprendizado padrão do otimizador.
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
	private static final double PADRAO_EPS = 1e-8;

	/**
	 * Valor de taxa de aprendizado do otimizador.
	 */
	private final double tA;

	/**
	 * Decaimento do momentum.
	 */
	private final double beta1;
	 
	/**
	 * Decaimento do momentum de segunda ordem.
	 */
	private final double beta2;
	 
	/**
	 * Usado para evitar divisão por zero.
	 */
	private final double eps;

	/**
	 * Coeficientes de momentum.
	 */
	private Tensor[] m;

	/**
	 * Coeficientes de momentum de segunda ordem.
	 */
	private Tensor[] v;
	
	/**
	 * Contador de iterações.
	 */
	long iteracoes = 0L;
 
	/**
	 * Inicializa uma nova instância de otimizador <strong> Adam </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA taxa de aprendizado do otimizador.
	 * @param beta1 decaimento do momento de primeira ordem.
	 * @param beta2 decaimento do momento de segunda ordem.
	 * @param eps usado para evitar a divisão por zero.
	 */
	public Adam(double tA, double beta1, double beta2, double eps) {
		if (tA <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de aprendizado (" + tA + ") inválida."
			);
		}
		if (beta1 <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de decaimento de primeira ordem (" + beta1 + ") inválida."
			);
		}
		if (beta2 <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de decaimento de segunda ordem (" + beta2 + ") inválida."
			);
		}
		if (eps <= 0) {
			throw new IllegalArgumentException(
				"\nEpsilon (" + eps + ") inválido."
			);
		}
		
		this.tA = tA;
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.eps = eps;
	}
 
	/**
	 * Inicializa uma nova instância de otimizador <strong> Adam </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA taxa de aprendizado do otimizador.
	 * @param beta1 decaimento do momento de primeira ordem.
	 * @param beta2 decaimento do momento de segunda ordem.
	 */
	public Adam(double tA, double beta1, double beta2) {
		this(tA, beta1, beta2, PADRAO_EPS);
	}
 
	/**
	 * Inicializa uma nova instância de otimizador <strong> Adam </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA taxa de aprendizado do otimizador.
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
	public void construir(Modelo modelo) {
		initParams(modelo);

		m = new Tensor[0];
		v = new Tensor[0];
		for (Tensor t : _params) {
			m = utils.addEmArray(m, new Tensor(t.shape()));
			v = utils.addEmArray(v, new Tensor(t.shape()));
		}

		_construido = true;// otimizador pode ser usado
	}

	@Override
	public void atualizar() {
		verificarConstrucao();
		
		iteracoes++;
		double fb1 = Math.pow(beta1, iteracoes);
		double fb2 = Math.pow(beta2, iteracoes);
		double alfa = tA * Math.sqrt(1 - fb2) / (1 - fb1);
		
		for (int i = 0; i < _params.length; i++) {
			m[i].aplicar(m[i], _grads[i], 
				(m, g) -> m + ((1 - beta1) * (g - m))
			);

			v[i].aplicar(v[i], _grads[i], 
				(v, g) -> v + ((1 - beta2) * ((g*g) - v))
			);

			_params[i].aplicar(_params[i], m[i], v[i], 
				(p, m, v) -> p - (((alfa * m) / (Math.sqrt(v) + eps)))
			);
		}
	}

	@Override
	public String info() {
		verificarConstrucao();
		construirInfo();
		
		addInfo("Lr: " + tA);
		addInfo("Beta1: " + beta1);
		addInfo("Beta2: " + beta2);
		addInfo("Epsilon: " + eps);

		return super.info();
	}

}
