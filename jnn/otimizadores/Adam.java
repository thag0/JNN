package jnn.otimizadores;

import jnn.core.tensor.Tensor;
import jnn.core.tensor.TensorData;

/**
 * <h2>
 *    Adaptive Moment Estimation
 * </h2>
 * Implementação do algoritmo de otimização Adam.
 * <p>
 *    O algoritmo ajusta os parâmetros do modelo usando o gradiente descendente 
 *    com momento e estimativas adaptativas para momentos de primeira e segunda ordem.
 * </p>
 * {@link {@code Paper}: https://arxiv.org/pdf/1412.6980}
 */
public class Adam extends Otimizador {

	/**
	 * Valor de taxa de aprendizado padrão do otimizador.
	 */
	static final double PADRAO_LR = 0.001;

	/**
	 * Valor padrão para o decaimento do momento de primeira ordem.
	 */
	static final double PADRAO_BETA1 = 0.9;
 
	/**
	 * Valor padrão para o decaimento do momento de segunda ordem.
	 */
	static final double PADRAO_BETA2 = 0.999;
	 
	/**
	 * Valor padrão para epsilon.
	 */
	static final double PADRAO_EPS = 1e-8;
	 
	/**
	 * Valor padrão de correção.
	 */
	static final boolean PADRAO_AMSGRAD = false;

	/**
	 * Valor de taxa de aprendizado do otimizador (Learning Rate).
	 */
	final double lr;

	/**
	 * Decaimento do momentum.
	 */
	final double beta1;
	 
	/**
	 * Decaimento do momentum de segunda ordem.
	 */
	final double beta2;
	 
	/**
	 * Usado para evitar divisão por zero.
	 */
	final double eps;

	/**
	 * Correção dos valores de velocidade.
	 */
	final boolean amsgrad;

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
	 *  Coeficientes de correção do AMSGrad.
	 */
	private Tensor[] ams = {};
	
	/**
	 * Contador de iterações.
	 */
	long iteracao = 0L;
 
	/**
	 * Inicializa uma nova instância de otimizador <strong> Adam </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param lr taxa de aprendizado do otimizador.
	 * @param beta1 decaimento do momento de primeira ordem.
	 * @param beta2 decaimento do momento de segunda ordem.
	 * @param eps pequeno valor usado para evitar a divisão por zero.
	 * @param amsgrad aplicar correção.
	 */
	public Adam(Number lr, Number beta1, Number beta2, Number eps, boolean amsgrad) {
		double lr_ = lr.doubleValue();
		double beta1_ = beta1.doubleValue();
		double beta2_ = beta2.doubleValue();
		double eps_ = eps.doubleValue();

		if (lr_ <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de aprendizado (" + lr_ + ") inválida."
			);
		}
		if (beta1_ <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de decaimento de primeira ordem (" + beta1_ + ") inválida."
			);
		}
		if (beta2_ <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de decaimento de segunda ordem (" + beta2_ + ") inválida."
			);
		}
		if (eps_ <= 0) {
			throw new IllegalArgumentException(
				"\nEpsilon (" + eps_ + ") inválido."
			);
		}
		
		this.lr 	 = lr_;
		this.beta1 	 = beta1_;
		this.beta2 	 = beta2_;
		this.eps 	 = eps_;
		this.amsgrad = amsgrad;
	}
 
	/**
	 * Inicializa uma nova instância de otimizador <strong> Adam </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param lr taxa de aprendizado do otimizador.
	 * @param beta1 decaimento do momento de primeira ordem.
	 * @param beta2 decaimento do momento de segunda ordem.
	 * @param eps pequeno valor usado para evitar a divisão por zero.
	 */
	public Adam(Number lr, Number beta1, Number beta2, Number eps) {
		this(lr, beta1, beta2, eps, PADRAO_AMSGRAD);
	}
 
	/**
	 * Inicializa uma nova instância de otimizador <strong> Adam </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param lr taxa de aprendizado do otimizador.
	 * @param beta1 decaimento do momento de primeira ordem.
	 * @param beta2 decaimento do momento de segunda ordem.
	 */
	public Adam(Number lr, Number beta1, Number beta2) {
		this(lr, beta1, beta2, PADRAO_EPS, PADRAO_AMSGRAD);
	}
 
	/**
	 * Inicializa uma nova instância de otimizador <strong> Adam </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param lr taxa de aprendizado do otimizador.
	 */
	public Adam(Number lr) {
		this(lr, PADRAO_BETA1, PADRAO_BETA2, PADRAO_EPS, PADRAO_AMSGRAD);
	}
 
	/**
	 * Inicializa uma nova instância de otimizador <strong> Adam </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param amsgrad aplicar correção.
	 */
	public Adam(boolean amsgrad) {
		this(PADRAO_LR, PADRAO_BETA1, PADRAO_BETA2, PADRAO_EPS, amsgrad);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> Adam </strong>.
	 * <p>
	 *    Os hiperparâmetros do Adam serão inicializados com os valores 
	 *    padrão.
	 * </p>
	 */
	public Adam() {
		this(PADRAO_LR, PADRAO_BETA1, PADRAO_BETA2, PADRAO_EPS, PADRAO_AMSGRAD);
	}

	@Override
	public void construir(Tensor[] params, Tensor[] grads) {
		initParams(params, grads);

		for (Tensor param : _params) {
			m = utils.addEmArray(m, new Tensor(param.shape()));
			v = utils.addEmArray(v, new Tensor(param.shape()));
			mc = utils.addEmArray(mc, new Tensor(param.shape()));
			vc = utils.addEmArray(vc, new Tensor(param.shape()));
		}

		if (amsgrad) {
			for (Tensor param : _params) {
				ams = utils.addEmArray(ams, new Tensor(param.shape()));
			}
		}
		
		_construido = true;// otimizador pode ser usado
	}

	@Override
	public void atualizar() {
		verificarConstrucao();
		
		iteracao += 1;

		double corr1 = Math.pow(beta1, iteracao);
		double corr2 = Math.pow(beta2, iteracao);

		final int n = _params.length;
		for (int i = 0; i < n; i++) {
			TensorData p_i = _params[i].data();
			TensorData g_i = _grads[i].data();
			TensorData m_i = m[i].data();
			TensorData v_i = v[i].data();
			TensorData mc_i = mc[i].data();
			TensorData vc_i = vc[i].data();

			// m = β1*m + (g * (1 - β1))
			m_i.mul(beta1).add(g_i, 1.0 - beta1);

			// v = β2*v + (g² * (1 - β2))
			v_i.mul(beta2).addcmul(g_i, g_i, 1.0 - beta2);

			// m̂ = m / (1 - β1^t)
			mc_i.copiar(m_i).div(1.0 - corr1);

			// v̂ = v / (1 - β2^t)
			vc_i.copiar(v_i).div(1.0 - corr2);

			if (amsgrad) {// vc = max(vc, vams)
				TensorData vams_i = ams[i].data();
				vams_i.maxEntre(vc_i);
				vc_i.copiar(vams_i);
			}

			// sqrt(v̂) + eps
			TensorData den = vc_i.clone().sqrt().add(eps);
			
			// p -= (lr * m̂) / (sqrt(v̂) + eps)
			p_i.addcdiv(mc_i, den, -lr);
		}
	}

	@Override
	public String info() {
		verificarConstrucao();
		construirInfo();
		
		addInfo("Lr: " + lr);
		addInfo("Beta1: " + beta1);
		addInfo("Beta2: " + beta2);
		addInfo("Epsilon: " + eps);
		addInfo("Amsgrad: " + amsgrad);

		return info();
	}

}
