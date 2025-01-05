package jnn.otimizadores;

import jnn.core.tensor.Tensor;

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
 * Caso a correção {@code amsgrad} esteja ativa:
 *<pre>
 *v = max(v, vc);
 *</pre>
 * Onde:
 * <p>
 *    {@code beta1 e beta2} - valores de decaimento dos momentums de primeira
 *    e segunda ordem.
 * </p>
 * <p>
 *    {@code g} - gradiente correspondente a variável que será otimizada.
 * </p>
 * <p>
 *    {@code vc} - histórico de atualizações do amsgrad.
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
	 * Valor padrão de correção.
	 */
	private static final boolean PADRAO_AMSGRAD = false;

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
	 * Correção dos valores de velocidade.
	 */
	private final boolean amsgrad;

	/**
	 * Coeficientes de momentum.
	 */
	private Tensor[] m = {};

	/**
	 * Coeficientes de momentum de segunda ordem.
	 */
	private Tensor[] v = {};

	/**
	 * Coeficientes de segunda ordem corrigidos.
	 */
	private Tensor[] vc;
	
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
	 * @param eps pequeno valor usado para evitar a divisão por zero.
	 * @param amsgrad aplicar correção.
	 */
	public Adam(Number tA, Number beta1, Number beta2, Number eps, boolean amsgrad) {
		double lr = tA.doubleValue();
		double b1 = beta1.doubleValue();
		double b2 = beta2.doubleValue();
		double ep = eps.doubleValue();

		if (lr <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de aprendizado (" + lr + ") inválida."
			);
		}
		if (b1 <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de decaimento de primeira ordem (" + b1 + ") inválida."
			);
		}
		if (b2 <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de decaimento de segunda ordem (" + b2 + ") inválida."
			);
		}
		if (ep <= 0) {
			throw new IllegalArgumentException(
				"\nEpsilon (" + ep + ") inválido."
			);
		}
		
		this.tA 	 = lr;
		this.beta1 	 = b1;
		this.beta2 	 = b2;
		this.eps 	 = ep;
		this.amsgrad = amsgrad;
	}
 
	/**
	 * Inicializa uma nova instância de otimizador <strong> Adam </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA taxa de aprendizado do otimizador.
	 * @param beta1 decaimento do momento de primeira ordem.
	 * @param beta2 decaimento do momento de segunda ordem.
	 * @param eps pequeno valor usado para evitar a divisão por zero.
	 */
	public Adam(Number tA, Number beta1, Number beta2, Number eps) {
		this(tA, beta1, beta2, eps, PADRAO_AMSGRAD);
	}
 
	/**
	 * Inicializa uma nova instância de otimizador <strong> Adam </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA taxa de aprendizado do otimizador.
	 * @param beta1 decaimento do momento de primeira ordem.
	 * @param beta2 decaimento do momento de segunda ordem.
	 */
	public Adam(Number tA, Number beta1, Number beta2) {
		this(tA, beta1, beta2, PADRAO_EPS, PADRAO_AMSGRAD);
	}
 
	/**
	 * Inicializa uma nova instância de otimizador <strong> Adam </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA taxa de aprendizado do otimizador.
	 */
	public Adam(Number tA) {
		this(tA, PADRAO_BETA1, PADRAO_BETA2, PADRAO_EPS, PADRAO_AMSGRAD);
	}
 
	/**
	 * Inicializa uma nova instância de otimizador <strong> Adam </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param amsgrad aplicar correção.
	 */
	public Adam(boolean amsgrad) {
		this(PADRAO_TA, PADRAO_BETA1, PADRAO_BETA2, PADRAO_EPS, amsgrad);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> Adam </strong>.
	 * <p>
	 *    Os hiperparâmetros do Adam serão inicializados com os valores 
	 *    padrão.
	 * </p>
	 */
	public Adam() {
		this(PADRAO_TA, PADRAO_BETA1, PADRAO_BETA2, PADRAO_EPS, PADRAO_AMSGRAD);
	}

	@Override
	public void construir(Tensor[] params, Tensor[] grads) {
		initParams(params, grads);

		for (Tensor param : _params) {
			m = utils.addEmArray(m, new Tensor(param.shape()));
			v = utils.addEmArray(v, new Tensor(param.shape()));
		}

		if (amsgrad) {
			vc = new Tensor[0];
			for (Tensor param : _params) {
				vc = utils.addEmArray(vc, new Tensor(param.shape()));
			}
		}
		
		_construido = true;// otimizador pode ser usado
	}

	@Override
	public void atualizar() {
		verificarConstrucao();
		
		iteracoes++;
		double fb1 = Math.pow(beta1, iteracoes);
		double fb2 = Math.pow(beta2, iteracoes);
		double alfa = tA * Math.sqrt(1.0 - fb2) / (1.0 - fb1);
		
		for (int i = 0; i < _params.length; i++) {
			m[i].aplicar(m[i], _grads[i], 
				(m, g) -> m + ((1.0 - beta1) * (g - m))
			);

			v[i].aplicar(v[i], _grads[i], 
				(v, g) -> v + ((1.0 - beta2) * ((g*g) - v))
			);

			if (amsgrad) {
				vc[i].aplicar(v[i], vc[i],
					(v, vc) -> Math.max(v, vc)
				);

				v[i].copiar(vc[i]);
			}

			_params[i].aplicar(_params[i], m[i], v[i], 
				(p, m, v) -> p - (((m * alfa) / (Math.sqrt(v) + eps)))
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
		addInfo("Amsgrad: " + amsgrad);

		return super.info();
	}

}
