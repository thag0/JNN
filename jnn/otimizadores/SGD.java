package jnn.otimizadores;

import jnn.core.tensor.Tensor;
import jnn.core.tensor.TensorData;

/**
 * <h2>
 *    Stochastic Gradient Descent 
 * </h2>
 * <p>
 *    Implementação do otimizador do gradiente estocástico com momentum e
 *    acelerador de nesterov.
 * </p>
 * <p>
 *    O SGD funciona usando a seguinte expressão:
 * </p>
 * <pre>
 *m += (m * M) - (g * tA)
 *v += m // apenas com momentum
 *v += (M * m) - (g * tA) // com nesterov
 * </pre>
 * Onde:
 * <p>
 *    {@code v} - variável que será otimizada.
 * </p>
 * <p>
 *    {@code M} - valor de taxa de momentum (ou constante de momentum) 
 *    do otimizador.
 * </p>
 * <p>
 *    {@code m} - valor de momentum da correspondente a variável que será
 *    otimizada.
 * </p>
 * <p>
 *    {@code g} - gradientes correspondente a variável que será otimizada.
 * </p>
 * <p>
 *    {@code tA} - taxa de aprendizado do otimizador.
 * </p>
 */
public class SGD extends Otimizador {

	/**
	 * Taxa de aprendizado padrão do otimizador.
	 */
	private static final double PADRAO_TA = 0.01;

	/**
	 * Taxa de momentum padrão do otimizador.
	 */
	private static final double PADRAO_MOMENTUM = 0.9;

	/**
	 * Uso do acelerador de nesterov padrão.
	 */
	private static final boolean PADRAO_NESTEROV = false;

	/**
	 * Valor de taxa de aprendizado do otimizador.
	 */
	private final double lr;

	/**
	 * Valor de taxa de momentum do otimizador.
	 */
	private final double momentum;

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
		double lr_ = lr.doubleValue();
		double mm_ = m.doubleValue();

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
		this(PADRAO_TA, PADRAO_MOMENTUM, PADRAO_NESTEROV);
	}

	@Override
	public void construir(Tensor[] params, Tensor[] grads) {
		initParams(params, grads);

		for (Tensor param : _params) {
			m = utils.addEmArray(m, new Tensor(param.shape()));
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
            TensorData m_i = m[i].data();

            // m = m * momentum - lr * g
            m_i.mul(momentum).add(g_i, -lr);

            if (nesterov) {
                // p += momentum * m - lr * grad
                p_i.add(m_i, momentum).add(g_i, -lr);
            } else {
                // p += m
                p_i.add(m_i);
            }
        }
	}

	@Override
	public String info() {
		verificarConstrucao();
		construirInfo();
		
		addInfo("Lr: " + lr);
		addInfo("Momentum: " + momentum);
		addInfo("Nesterov: " + nesterov);

		return super.info();
	}

}
