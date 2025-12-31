package jnn.otm;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;
import jnn.core.tensor.TensorData;

/**
 * <h2>
 *    Root Mean Square Propagation
 * </h2>
 * Implementação do algoritmo de otimização RMSProp.
 * <p>
 *    Ele é uma adaptação do Gradiente Descendente Estocástico (SGD) que ajuda a lidar com a
 *    oscilação do gradiente, permitindo que a taxa de aprendizado seja adaptada para cada 
 *    parâmetro individualmente.
 * </p>
 * {@link {@code Paper}: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf}
 */
public class RMSProp extends Otimizador {

	/**
	 * Valor padrão para a taxa de aprendizagem do otimizador.
	 */
	private static final double PADRAO_LR  = 0.001;

	/**
	 * Valor padrão para a taxa de decaimeto.
	 */
	private static final double PADRAO_RHO = 0.99;

	/**
	 * Valor padrão para epsilon.
	 */
	private static final double PADRAO_EPS = 1e-8;

	/**
	 * Valor de taxa de aprendizagem do otimizador (Learning Rate).
	 */
	private final double lr;

	/**
	 * Usado para evitar divisão por zero.
	 */
	private final double eps;

	/**
	 * Fator de decaimento.
	 */
	private final double rho;

	/**
	 * Acumuladores para os.
	 */
	private Tensor[] ac = {};

	/**
	 * Inicializa uma nova instância de otimizador <strong> RMSProp </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param lr valor de taxa de aprendizagem.
	 * @param rho fator de decaimento do RMSProp.
	 * @param eps usado para evitar a divisão por zero.
	 */
	public RMSProp(Number lr, Number rho, Number eps) {
		double lr_ = lr.doubleValue();
		double rho_ = rho.doubleValue();
		double eps_ = eps.doubleValue();

		if (lr_ <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de aprendizagem (" + lr_ + "), inválida."
			);
		}
		if (rho_ <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de decaimento (" + rho_ + "), inválida."
			);
		}
		if (eps_ <= 0) {
			throw new IllegalArgumentException(
				"\nEpsilon (" + eps_ + "), inválido."
			);
		}

		this.lr  = lr_;
		this.rho = rho_;
		this.eps = eps_;
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> RMSProp </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param lr valor de taxa de aprendizagem.
	 * @param rho fator de decaimento do RMSProp.
	 */
	public RMSProp(Number lr, Number rho) {
		this(lr, rho, PADRAO_EPS);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> RMSProp </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param lr valor de taxa de aprendizagem.
	 */
	public RMSProp(Number lr) {
		this(lr, PADRAO_RHO, PADRAO_EPS);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> RMSProp </strong>.
	 * <p>
	 *    Os hiperparâmetros do RMSProp serão inicializados com os valores padrão.
	 * </p>
	 */
	public RMSProp() {
		this(PADRAO_LR, PADRAO_RHO, PADRAO_EPS);
	}

	@Override
	public void construir(Tensor[] params, Tensor[] grads) {
		initParams(params, grads);

		for (Tensor param : _params) {
			ac = JNNutils.addEmArray(ac, new Tensor(param.shape()));
		}
		
		_construido = true;// otimizador pode ser usado
	}

	@Override
	public void update() {
		checkInicial();
		
		final int n = _params.length;
		for (int i = 0; i < n; i++) {
			TensorData p_i  = _params[i].data();
			TensorData g_i  = _grads[i].data();
			TensorData ac_i = ac[i].data();

			// ac = (rho * ac) + ((1 - rho) * g²)
			ac_i.mul(rho).addcmul(g_i, g_i, 1.0 - rho);

			// sqrt(ac) + eps
			TensorData den = ac_i.clone().sqrt().add(eps);

			// p -= (lr * g) / (sqrt(ac) + eps)
			p_i.addcdiv(g_i, den, -lr);
		}
	}

	@Override
	public String info() {
		checkInicial();
		construirInfo();
		
		addInfo("Lr: " + lr);
		addInfo("Rho: " + rho);
		addInfo("Epsilon: " + eps);

		return super.info();
	}

}
