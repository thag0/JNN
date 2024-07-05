package jnn.otimizadores;

import jnn.camadas.Camada;
import jnn.core.tensor.Tensor;

/**
 * <h2>
 *    Nesterov-accelerated Adaptive Moment Estimation
 * </h2>
 * Implementação do algoritmo de otimização Nadam.
 * <p>
 *    O algoritmo ajusta os pesos da rede neural usando o gradiente descendente 
 *    com momento e a estimativa adaptativa de momentos de primeira e segunda ordem.
 * </p>
 * O adicional do Nadam é usar o acelerador de nesterov durante a correção dos
 * pesos da rede.
 * <p>
 *    O Nadam funciona usando a seguinte expressão:
 * </p>
 * <pre>
 *    v -= (tA * mc) / ((√ vc) + eps)
 * </pre>
 * Onde:
 * <p>
 *    {@code v} - variável que será otimizada.
 * </p>
 * <p>
 *    {@code tA} - valor de taxa de aprendizado do otimizador.
 * </p>
 * <p>
 *    {@code mc} - valor de momentum corrigido
 * </p>
 * <p>
 *    {@code vc} - valor de velocidade (momentum de segunda ordem) corrigido
 * </p>
 * Os valores de momentum e velocidade corrigidos se dão por:
 * <pre>
 *mc = ((beta1 * m) + ((1 - beta1) * g)) / (1 - beta1ⁱ)
 *vc = (beta2 * v) / (1 - beta2ⁱ)
 * </pre>
 * Onde:
 * <p>
 *    {@code m} - valor de momentum correspondente a variável que será otimizada.
 * </p>
 * <p>
 *    {@code v} - valor de velocidade correspondente a variável que será otimizada.
 * </p>
 * <p>
 *    {@code g} - gradiente correspondente a variável que será otimizada.
 * </p>
 * <p>
 *    {@code i} - contador de interações do otimizador.
 * </p>
 */
public class Nadam extends Otimizador {

	/**
	 * Valor padrão para a taxa de aprendizado do otimizador.
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
	 * Usado para evitar divisão por zero.
	 */
	private final double eps;

	/**
	 * decaimento do momentum.
	 */
	private final double beta1;

	/**
	 * decaimento do momentum de segunda ordem.
	 */
	private final double beta2;

	/**
	 * Coeficientes de momentum.
	 */
	private Tensor[] m;

	/**
	 * Coeficientes de momentum de segunda ordem.
	 */
	private Tensor[] v;

	/**
	 * Coeficientes de momentum corrigidos.
	 */
	private Tensor[] mc;

	/**
	 * Coeficientes de momentum de segunda ordem corrigidos.
	 */
	private Tensor[] vc;

	/**
	 * Contador de iterações.
	 */
	long iteracoes = 0;

	/**
	 * Inicializa uma nova instância de otimizador <strong> Nadam </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA valor de taxa de aprendizado.
	 * @param beta1 decaimento do momento de primeira ordem.
	 * @param beta2 decaimento da segunda ordem.
	 * @param eps usado para evitar a divisão por zero.
	 */
	public Nadam(double tA, double beta1, double beta2, double eps) {
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
				"\nEpsilon (" + eps + "), inválido."
			);
		}
		
		this.tA = tA;
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.eps = eps;
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> Nadam </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA valor de taxa de aprendizado.
	 * @param beta1 decaimento do momento de primeira ordem.
	 * @param beta2 decaimento da segunda ordem.
	 */
	public Nadam(double tA, double beta1, double beta2) {
		this(tA, beta1, beta2, PADRAO_EPS);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> Nadam </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA valor de taxa de aprendizado.
	 */
	public Nadam(double tA) {
		this(tA, PADRAO_BETA1, PADRAO_BETA2, PADRAO_EPS);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> Nadam </strong>.
	 * <p>
	 *    Os hiperparâmetros do Nadam serão inicializados com os valores padrão.
	 * </p>
	 */
	public Nadam() {
		this(PADRAO_TA, PADRAO_BETA1, PADRAO_BETA2, PADRAO_EPS);
	}

	@Override
	public void construir(Camada[] camadas) {
		initParams(camadas);

		m  = new Tensor[0];
		v  = new Tensor[0];
		mc = new Tensor[0];
		vc = new Tensor[0];
		for (Tensor t : _params) {
			m  = utils.addEmArray(m,  new Tensor(t.shape()));
			v  = utils.addEmArray(v,  new Tensor(t.shape()));
			mc = utils.addEmArray(mc, new Tensor(t.shape()));
			vc = utils.addEmArray(vc, new Tensor(t.shape()));
		}

		_construido = true;// otimizador pode ser usado
	}

	@Override
	public void atualizar() {
		verificarConstrucao();
		
		iteracoes++;
		double fb1 = 1 - Math.pow(beta1, iteracoes);
		double fb2 = 1 - Math.pow(beta2, iteracoes);
		
		for (int i = 0; i < _params.length; i++) {
			m[i].aplicar(m[i], _grads[i], 
				(m, g) -> (beta1 * m) + ((1 - beta1) * g)
			);
			v[i].aplicar(v[i], _grads[i], 
				(v, g) -> (beta2 * v) + ((1 - beta2) * (g*g))
			);
			mc[i].aplicar(m[i], _grads[i],
				(m, g) -> (beta1 * m) + ((1 - beta1) * g) / fb1
			);
			vc[i].aplicar(v[i], _grads[i],
				(v, g) -> (beta2 * v) / fb2
			);
			_params[i].aplicar(_params[i], mc[i], vc[i], 
				(p, mc, vc) -> p -= (mc * tA) / (Math.sqrt(vc) + eps)
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
