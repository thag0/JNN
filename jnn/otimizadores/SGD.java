package jnn.otimizadores;

import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;

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
	private final double tA;

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
	private Tensor[] m;

	/**
	 * Inicializa uma nova instância de otimizador <strong> Stochastic Gradient 
	 * Descent (SGD) </strong> usando os valores de hiperparâmetros fornecidos.
	 * @param tA taxa de aprendizado do otimizador.
	 * @param m taxa de momentum do otimizador.
	 * @param nesterov usar acelerador de nesterov.
	 */
	public SGD(double tA, double m, boolean nesterov) {
		if (tA <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de aprendizado (" + tA + ") inválida."
			);
		}
		
		if (m < 0) {         
			throw new IllegalArgumentException(
				"\nTaxa de momentum (" + m + ") inválida."
			);
		}

		this.tA = tA;
		this.momentum = m;
		this.nesterov = nesterov;
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> Stochastic Gradient 
	 * Descent (SGD) </strong> usando os valores de hiperparâmetros fornecidos.
	 * @param tA taxa de aprendizado do otimizador.
	 * @param m taxa de momentum do otimizador.
	 */
	public SGD(double tA, double m) {
		this(tA, m, PADRAO_NESTEROV);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> Stochastic Gradient 
	 * Descent (SGD) </strong> usando os valores de hiperparâmetros fornecidos.
	 * @param tA taxa de aprendizado do otimizador.
	 */
	public SGD(double tA) {
		this(tA, PADRAO_MOMENTUM, PADRAO_NESTEROV);
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
	public void construir(Modelo modelo) {
		initParams(modelo);

		m = new Tensor[0];
		for (Tensor t : _params) {
			m = utils.addEmArray(m, new Tensor(t.shape()));
		}
		
		_construido = true;// otimizador pode ser usado
	}
	
	@Override
	public void atualizar() {
		verificarConstrucao();
		
		for (int i = 0; i < _params.length; i++) {
			m[i].aplicar(m[i], _grads[i], 
				(m, g) -> (m * momentum) - (g * tA)
			);

			if (nesterov) {
				_params[i].aplicar(_params[i], m[i], _grads[i], 
					(p, m, g) -> p + (momentum * m) - (g * tA)
				);
			} else {
				_params[i].add(m[i]);
			}
		}
	}

	@Override
	public String info() {
		verificarConstrucao();
		construirInfo();
		
		addInfo("Lr: " + tA);
		addInfo("Momentum: " + momentum);
		addInfo("Nesterov: " + nesterov);

		return super.info();
	}

}
