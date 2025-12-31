package jnn.otm;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;
import jnn.core.tensor.TensorData;

/**
 * <h2>
 *    Nesterov-accelerated Adaptive Moment Estimation
 * </h2>
 * Implementação do algoritmo de otimização Nadam.
 * <p>
 *		O algoritmo ajusta os pesos da rede neural usando o gradiente descendente 
 *		com momento e a estimativa adaptativa de momentos de primeira e segunda ordem
 *		tendo como adicionar o acelerador de nesterov para correção.
 * </p>
 * {@link {@code Paper} https://cs229.stanford.edu/proj2015/054_report.pdf}
 */
public class Nadam extends Otimizador {

	/**
	 * Valor padrão para a taxa de aprendizado do otimizador.
	 */
	private static final double PADRAO_LR = 0.001;

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
	private final double lr;

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
	private Tensor[] m = {};

	/**
	 * Coeficientes de momentum de segunda ordem.
	 */
	private Tensor[] v = {};

	/**
	 * Coeficientes de momentum corrigidos.
	 */
	private Tensor[] mc = {};

	/**
	 * Coeficientes de momentum de segunda ordem corrigidos.
	 */
	private Tensor[] vc = {};

	/**
	 * Contador de iterações.
	 */
	long iteracoes = 0;

	/**
	 * Inicializa uma nova instância de otimizador <strong> Nadam </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param lr valor de taxa de aprendizado.
	 * @param beta1 decaimento do momento de primeira ordem.
	 * @param beta2 decaimento da segunda ordem.
	 * @param eps usado para evitar a divisão por zero.
	 */
	public Nadam(Number lr, Number beta1, Number beta2, Number eps) {
		double lr_ = lr.doubleValue();
		double b1_ = beta1.doubleValue();
		double b2_ = beta2.doubleValue();
		double ep_ = eps.doubleValue();

		if (lr_ <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de aprendizado (" + lr_ + ") inválida."
			);
		}
		if (b1_ <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de decaimento de primeira ordem (" + b1_ + ") inválida."
			);
		}
		if (b2_ <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de decaimento de segunda ordem (" + b2_ + ") inválida."
			);
		}
		if (ep_ <= 0) {
			throw new IllegalArgumentException(
				"\nEpsilon (" + ep_ + ") inválido."
			);
		}
		
		this.lr 	 = lr_;
		this.beta1 	 = b1_;
		this.beta2 	 = b2_;
		this.eps 	 = ep_;
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> Nadam </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param lr valor de taxa de aprendizado.
	 * @param beta1 decaimento do momento de primeira ordem.
	 * @param beta2 decaimento da segunda ordem.
	 */
	public Nadam(Number lr, Number beta1, Number beta2) {
		this(lr, beta1, beta2, PADRAO_EPS);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> Nadam </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param lr valor de taxa de aprendizado.
	 */
	public Nadam(Number lr) {
		this(lr, PADRAO_BETA1, PADRAO_BETA2, PADRAO_EPS);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> Nadam </strong>.
	 * <p>
	 *    Os hiperparâmetros do Nadam serão inicializados com os valores padrão.
	 * </p>
	 */
	public Nadam() {
		this(PADRAO_LR, PADRAO_BETA1, PADRAO_BETA2, PADRAO_EPS);
	}

	@Override
	public void construir(Tensor[] params, Tensor[] grads) {
		initParams(params, grads);
		
		for (Tensor param : _params) {
			int[] shape = param.shape();

			m  = JNNutils.addEmArray(m,  new Tensor(shape));
			v  = JNNutils.addEmArray(v,  new Tensor(shape));
			mc = JNNutils.addEmArray(mc, new Tensor(shape));
			vc = JNNutils.addEmArray(vc, new Tensor(shape));
		}

		_construido = true;// otimizador pode ser usado
	}

	@Override
	public void update() {
		checkInicial();
		
		iteracoes++;
		double fb1 = 1.0 - Math.pow(beta1, iteracoes);
		double fb2 = 1.0 - Math.pow(beta2, iteracoes);
		
		final int n = _params.length;
		for (int i = 0; i < n; i++) {
			TensorData p_i  = _params[i].data();
			TensorData g_i  = _grads[i].data();
			TensorData m_i  = m[i].data();
			TensorData v_i  = v[i].data();
			TensorData mc_i  = mc[i].data();
			TensorData vc_i  = vc[i].data();

			// m = β1*m + (1-β1)*g
			m_i.mul(beta1).add(g_i, 1.0 - beta1);

			// v = β2*v + (1-β2)*(g²)
			v_i.mul(beta2).addcmul(g_i, g_i, 1.0 - beta2);

			// m̂ = (β1 * m + (1-β1)*g) / (1 - β1^t)
        	mc_i.copiar(m_i).mul(beta1).add(g_i, 1.0 - beta1).div(fb1);
			
			// v̂ = v/(1-β2^t)
			vc_i.copiar(v_i).div(fb2);

			// den = sqrt(v̂) + eps
			TensorData den = vc_i.sqrt().add(eps);

			// p -= (lr * m̂) / (sqrt(v̂) + eps)
			p_i.addcdiv(mc_i, den, -lr);
		}
	}

	@Override
	public String info() {
		checkInicial();
		construirInfo();
		
		addInfo("Lr: " + lr);
		addInfo("Beta1: " + beta1);
		addInfo("Beta2: " + beta2);
		addInfo("Epsilon: " + eps);

		return super.info();
	}

}
