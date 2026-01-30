package jnn.otm;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;
import jnn.core.tensor.TensorData;

/**
 * <h2>
 *    Stochastic Gradient Descent 
 * </h2>
 * Implementação do algoritmo de otimização SGD.
 * <p>
 *    Implementação do otimizador do gradiente estocástico com momentum e
 *    acelerador de nesterov.
 * </p>
 */
public class SGD extends Otimizador {

	/**
	 * Taxa de aprendizado padrão do otimizador.
	 */
	private static final float PADRAO_LR = 0.01f;

	/**
	 * Taxa de momentum padrão do otimizador.
	 */
	private static final float PADRAO_MOMENTUM = 0.9f;

	/**
	 * Uso do acelerador de nesterov padrão.
	 */
	private static final boolean PADRAO_NESTEROV = false;

	/**
	 * Valor de taxa de aprendizado do otimizador.
	 */
	private final float lr;

	/**
	 * Valor de taxa de momentum do otimizador.
	 */
	private final float momentum;

	/**
	 * Usar acelerador de Nesterov.
	 */
	private final boolean nesterov;

	/**
	 * Coeficientes de momentum.
	 */
	private Tensor[] m = {};

	/**
	 * Inicializa uma nova instância de otimizador <strong> Stochastic Gradient 
	 * Descent (SGD) </strong> usando os valores de hiperparâmetros fornecidos.
	 * @param lr taxa de aprendizado do otimizador.
	 * @param m taxa de momentum do otimizador.
	 * @param nesterov usar acelerador de nesterov.
	 */
	public SGD(Number lr, Number m, boolean nesterov) {
		float lr_ = lr.floatValue();
		float mm_ = m.floatValue();

		if (lr_ <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de aprendizado (" + lr + ") inválida."
			);
		}
		if (mm_ < 0) {         
			throw new IllegalArgumentException(
				"\nTaxa de momentum (" + mm_ + ") inválida."
			);
		}

		this.lr 	  = lr_;
		this.momentum = mm_;
		this.nesterov = nesterov;
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> Stochastic Gradient 
	 * Descent (SGD) </strong> usando os valores de hiperparâmetros fornecidos.
	 * @param lr taxa de aprendizado do otimizador.
	 * @param m taxa de momentum do otimizador.
	 */
	public SGD(Number lr, Number m) {
		this(lr, m, PADRAO_NESTEROV);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> Stochastic Gradient 
	 * Descent (SGD) </strong> usando os valores de hiperparâmetros fornecidos.
	 * @param lr taxa de aprendizado do otimizador.
	 */
	public SGD(Number lr) {
		this(lr, PADRAO_MOMENTUM, PADRAO_NESTEROV);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> Stochastic Gradient 
	 * Descent (SGD) </strong>.
	 * <p>
	 *    Os hiperparâmetros do SGD serão inicializados com seus os valores padrão.
	 * </p>
	 */
	public SGD() {
		this(PADRAO_LR, PADRAO_MOMENTUM, PADRAO_NESTEROV);
	}

	@Override
	public void construir(Tensor[] params, Tensor[] grads) {
		initParams(params, grads);

		for (Tensor param : _params) {
			m = JNNutils.addEmArray(m, new Tensor(param.shape()));
		}
		
		_construido = true;// otimizador pode ser usado
	}
	
	@Override
	public void update() {
		checkInicial();
		
        final int n = _params.length;
        for (int i = 0; i < n; i++) {
            TensorData p_i = _params[i].data();
            TensorData g_i = _grads[i].data();
            TensorData m_i = m[i].data();

            // m = (m * momentum) - (g * lr)
            m_i.mul(momentum).add(g_i, -lr);

            if (nesterov) {
				// p += (momentum * m) - (g * lr)
				p_i.add(m_i, momentum).add(g_i, -lr);
			} else {
				// p += m
				p_i.add(m_i);
			}
        }
	}

	@Override
	public String info() {
		checkInicial();
		construirInfo();
		
		addInfo("Lr: " + lr);
		addInfo("Momentum: " + momentum);
		addInfo("Nesterov: " + nesterov);

		return super.info();
	}

}
