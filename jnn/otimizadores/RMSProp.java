package jnn.otimizadores;

import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;

/**
 * <h2>
 *    Root Mean Square Propagation
 * </h2>
 * <p>
 *    Ele é uma adaptação do Gradiente Descendente Estocástico (SGD) que ajuda a lidar com a
 *    oscilação do gradiente, permitindo que a taxa de aprendizado seja adaptada para cada 
 *    parâmetro individualmente.
 * </p>
 * <p>
 * 	Os hiperparâmetros do RMSProp podem ser ajustados para controlar 
 *    o comportamento do otimizador durante o treinamento.
 * </p>
 * <p>
 *    O RMSProp funciona usando a seguinte expressão:
 * </p>
 * <pre>
 *ac = (rho * ac) + ((1- rho) * g²);
 *v -= (g * tA) / ((√ ac) + eps)
 * </pre>
 * Onde:
 * <p>
 *    {@code v} - variável que será otimizada..
 * </p>
 * <p>
 *    {@code g} - gradiente correspondente a variável
 *    que será otimizada.
 * </p>
 * <p>
 *    {@code tA} - taxa de aprendizagem do otimizador.
 * </p>
 * <p>
 *    {@code ac} - acumulador de gradiente correspondente a variável
 *    que será otimizada.
 * </p>
 * <p>
 *    {@code rho} - taxa de decaimento do otimizador.
 * </p>
 */
public class RMSProp extends Otimizador {

	/**
	 * Valor padrão para a taxa de aprendizagem do otimizador.
	 */
	private static final double PADRAO_TA  = 0.001;

	/**
	 * Valor padrão para a taxa de decaimeto.
	 */
	private static final double PADRAO_RHO = 0.995;

	/**
	 * Valor padrão para epsilon.
	 */
	private static final double PADRAO_EPS = 1e-8;

	/**
	 * Valor de taxa de aprendizagem do otimizador.
	 */
	private final double tA;

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
	private Tensor[] ac;

	/**
	 * Inicializa uma nova instância de otimizador <strong> RMSProp </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA valor de taxa de aprendizagem.
	 * @param rho fator de decaimento do RMSProp.
	 * @param eps usado para evitar a divisão por zero.
	 */
	public RMSProp(Number tA, Number rho, Number eps) {
		double lr = tA.doubleValue();
		double rh = rho.doubleValue();
		double ep = eps.doubleValue();

		if (lr <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de aprendizagem (" + lr + "), inválida."
			);
		}
		if (rh <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de decaimento (" + rh + "), inválida."
			);
		}
		if (ep <= 0) {
			throw new IllegalArgumentException(
				"\nEpsilon (" + ep + "), inválido."
			);
		}

		this.tA  = lr;
		this.rho = rh;
		this.eps = ep;
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> RMSProp </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA valor de taxa de aprendizagem.
	 * @param rho fator de decaimento do RMSProp.
	 */
	public RMSProp(Number tA, Number rho) {
		this(tA, rho, PADRAO_EPS);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> RMSProp </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA valor de taxa de aprendizagem.
	 */
	public RMSProp(Number tA) {
		this(tA, PADRAO_RHO, PADRAO_EPS);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> RMSProp </strong>.
	 * <p>
	 *    Os hiperparâmetros do RMSProp serão inicializados com os valores padrão.
	 * </p>
	 */
	public RMSProp() {
		this(PADRAO_TA, PADRAO_RHO, PADRAO_EPS);
	}

	@Override
	public void construir(Modelo modelo) {
		initParams(modelo);

		ac = new Tensor[0];
		for (Tensor param : _params) {
			ac = utils.addEmArray(ac, new Tensor(param.shape()));
		}
		
		_construido = true;// otimizador pode ser usado
	}

	@Override
	public void atualizar() {
		verificarConstrucao();
		
		for (int i = 0; i < _params.length; i++) {
			ac[i].aplicar(ac[i], _grads[i], 
				(ac, g) -> (rho * ac) + (1 - rho) * (g*g)
			);
			_params[i].aplicar(_params[i], _grads[i], ac[i], 
				(p, g, ac) -> p -= (g * tA) / (Math.sqrt(ac) + eps)
			);
		}
	}

	@Override
	public String info() {
		verificarConstrucao();
		construirInfo();
		
		addInfo("Lr: " + tA);
		addInfo("Rho: " + rho);
		addInfo("Epsilon: " + eps);

		return super.info();
	}

}
