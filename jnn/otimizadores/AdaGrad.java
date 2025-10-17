package jnn.otimizadores;

import jnn.core.tensor.Tensor;
import jnn.core.tensor.TensorData;

/**
 * <h2>
 *    Adaptive Gradient Algorithm
 * </h2>
 * Implementa uma versão do algoritmo AdaGrad (Adaptive Gradient Algorithm).
 * O algoritmo otimiza o processo de aprendizado adaptando a taxa de aprendizagem 
 * de cada parâmetro com base no histórico de atualizações 
 * anteriores.
 * <p>
 *    Devido a natureza do otimizador, pode ser mais vantajoso (para este caso específico)
 *    usar valores de taxa de aprendizagem mais altos.
 * </p>
 * {@link {@code Paper}: http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf}
 */
public class AdaGrad extends Otimizador {

	/**
	 * Valor padrão para a taxa de aprendizado do otimizador.
	 */
	private static final double PADRAO_LR = 0.5;

	/**
	 * Valor padrão para o valor de epsilon pro otimizador.
	 */
	private static final double PADRAO_EPS = 1e-8; 

	/**
	 * Valor de taxa de aprendizado do otimizador (Learning Rate).
	 */
	private final double lr;

	/**
	 * Usado para evitar divisão por zero.
	 */
	private final double eps;

	/**
	 * Acumuladores.
	 */
	private Tensor[] ac = {};

	/**
	 * Inicializa uma nova instância de otimizador <strong> AdaGrad </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param lr valor de taxa de aprendizado.
	 * @param eps usado para evitar a divisão por zero.
	 */
	public AdaGrad(Number lr, Number eps) {
		double lr_ = lr.doubleValue();
		double eps_ = eps.doubleValue();

		if (lr_ <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de aprendizado (" + lr_ + ") inválida."
			);
		}

		if (eps_ <= 0) {
			throw new IllegalArgumentException(
				"\nEpsilon (" + eps + ") inválido."
			);
		}
		
		this.lr  = lr_;
		this.eps = eps_;
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> AdaGrad </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param lr valor de taxa de aprendizado.
	 */
	public AdaGrad(Number lr) {
		this(lr, PADRAO_EPS);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> AdaGrad </strong>.
	 * <p>
	 *    Os hiperparâmetros do AdaGrad serão inicializados com os valores padrão.
	 * </p>
	 */
	public AdaGrad() {
		this(PADRAO_LR, PADRAO_EPS);
	}

	@Override
	public void construir(Tensor[] params, Tensor[] grads) {
		initParams(params, grads);
		
		double valorInicial = 0.1;
		for (Tensor param : _params) {
			ac = utils.addEmArray(
				ac,
				new Tensor(param.shape()).preencher(valorInicial)
			);
		}
		
		_construido = true;// otimizador pode ser usado
	}

	@Override
	public void atualizar() {
		verificarConstrucao();
		
		final int n = _params.length;
		for (int i = 0; i < n; i++) {
			TensorData p_i = _params[i].data();
			TensorData g_i = _grads[i].data();
			TensorData ac_i = ac[i].data();

			// ac += g²
			ac_i.addcmul(g_i, g_i, 1.0);

			// sqrt(ac) + eps
			TensorData den = ac_i.clone().sqrt().add(eps);

			// p -= (lr * g) / (sqrt(ac) + eps)
			p_i.addcdiv(g_i, den, -lr);
		}
	}

	@Override
	public String info() {
		verificarConstrucao();
		construirInfo();
		
		addInfo("Lr: " + lr);
		addInfo("Epsilon: " + eps);

		return super.info();
	}
}
