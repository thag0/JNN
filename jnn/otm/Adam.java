package jnn.otm;

import jnn.core.JNNutils;
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
 * @see <a href="https://arxiv.org/pdf/1412.6980"> Paper Adam </a>
 */
public class Adam extends Otimizador {

	/**
	 * Valor de taxa de aprendizado padrão do otimizador.
	 */
	static final float PADRAO_LR = 0.001f;

	/**
	 * Valor padrão para o decaimento do momento de primeira ordem.
	 */
	static final float PADRAO_BETA1 = 0.9f;
 
	/**
	 * Valor padrão para o decaimento do momento de segunda ordem.
	 */
	static final float PADRAO_BETA2 = 0.999f;
	 
	/**
	 * Valor padrão para epsilon.
	 */
	static final float PADRAO_EPS = 1e-7f;
	 
	/**
	 * Valor padrão de correção.
	 */
	static final boolean PADRAO_AMSGRAD = false;

	/**
	 * Valor de taxa de aprendizado do otimizador (Learning Rate).
	 */
	private float lr;

	/**
	 * Decaimento do momentum.
	 */
	private final float beta1;
	 
	/**
	 * Decaimento do momentum de segunda ordem.
	 */
	private final float beta2;
	 
	/**
	 * Usado para evitar divisão por zero.
	 */
	private final float eps;

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
	 *  Buffers pra evitar alocação.
	 */
	private Tensor[] buf = {};
	
	/**
	 * Contador de iterações.
	 */
	long iteracao = 0L;

	/**
	 * Valor cacheado para evitar math.pow.
	 */
	private float potBeta1 = 1;
	
	/**
	 * Valor cacheado para evitar math.pow.
	 */
	private float potBeta2 = 1;
 
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
		float lr_ = lr.floatValue();
		float beta1_ = beta1.floatValue();
		float beta2_ = beta2.floatValue();
		float eps_ = eps.floatValue();

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
			int[] shape = param.shape();

			m = JNNutils.addEmArray(m, new Tensor(shape));
			v = JNNutils.addEmArray(v, new Tensor(shape));
			mc = JNNutils.addEmArray(mc, new Tensor(shape));
			vc = JNNutils.addEmArray(vc, new Tensor(shape));
			buf = JNNutils.addEmArray(buf, new Tensor(shape));
		}

		if (amsgrad) {
			for (Tensor param : _params) {
				ams = JNNutils.addEmArray(ams, new Tensor(param.shape()));
			}
		}
		
		_construido = true;// otimizador pode ser usado
	}

	@Override
	public void update() {
		checkInicial();
		
		iteracao += 1;

		potBeta1 *= beta1;
		potBeta2 *= beta2;

		float corr1 = potBeta1;
		float corr2 = potBeta2;

		final int n = _params.length;
		for (int i = 0; i < n; i++) {
			TensorData p_i = _params[i].data();
			TensorData g_i = _grads[i].data();
			TensorData m_i = m[i].data();
			TensorData v_i = v[i].data();
			TensorData mc_i = mc[i].data();
			TensorData vc_i = vc[i].data();
			TensorData buf_i = buf[i].data();

			// m = β1*m + (g * (1 - β1))
			m_i.mul(beta1).add(g_i, 1.0f - beta1);

			// v = β2*v + (g² * (1 - β2))
			v_i.mul(beta2).addcmul(g_i, g_i, 1.0f - beta2);

			// m̂ = m / (1 - β1^t)
			mc_i.copiar(m_i).div(1.0f - corr1);

			// v̂ = v / (1 - β2^t)
			vc_i.copiar(v_i).div(1.0f - corr2);

			if (amsgrad) {// vc = max(vc, vams)
				TensorData vams_i = ams[i].data();
				vams_i.maxEntre(vc_i);
				vc_i.copiar(vams_i);
			}
			
			// den = sqrt(v̂) + eps
			buf_i.copiar(vc_i).sqrt().add(eps);
			
			// p -= (lr * m̂) / (sqrt(v̂) + eps)
			p_i.addcdiv(mc_i, buf_i, -lr);
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
		addInfo("Amsgrad: " + amsgrad);

		return super.info();
	}

	@Override
	public float getLr() {
		return lr;
	}

	@Override
	public void setLr(float lr) {
		if (lr <= 0) {
			throw new IllegalArgumentException("\nLearning rate \""+ lr + "\" inválido.");
		}

		this.lr = lr;
	}

}
