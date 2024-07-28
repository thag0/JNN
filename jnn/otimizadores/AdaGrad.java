package jnn.otimizadores;

import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;

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
 * <p>
 *    O Adagrad funciona usando a seguinte expressão:
 * </p>
 * <pre>
 *    v -= (tA * g) / (√ ac + eps)
 * </pre>
 * Onde:
 * <p>
 *    {@code v} - variável que será otimizada.
 * </p>
 * <p>
 *    {@code tA} - taxa de aprendizagem do otimizador.
 * </p>
 * <p>
 *    {@code g} - gradientes correspondente a variável que será otimizada.
 * </p>
 * <p>
 *    {@code ac} - acumulador de gradiente correspondente a variável que
 *    será otimizada
 * </p>
 * <p>
 *    {@code eps} - um valor pequeno para evitar divizões por zero.
 * </p>
 */
public class AdaGrad extends Otimizador {

	/**
	 * Valor padrão para a taxa de aprendizado do otimizador.
	 */
	private static final double PADRAO_TA = 0.5;

	/**
	 * Valor padrão para o valor de epsilon pro otimizador.
	 */
	private static final double PADRAO_EPS = 1e-7; 

	/**
	 * Valor de taxa de aprendizado do otimizador.
	 */
	private final double tA;

	/**
	 * Usado para evitar divisão por zero.
	 */
	private final double eps;

	/**
	 * Acumuladores.
	 */
	private Tensor[] ac;

	/**
	 * Inicializa uma nova instância de otimizador <strong> AdaGrad </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA valor de taxa de aprendizado.
	 * @param eps usado para evitar a divisão por zero.
	 */
	public AdaGrad(Number tA, Number eps) {
		double lr = tA.doubleValue();
		double e = eps.doubleValue();

		if (lr <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de aprendizado (" + lr + ") inválida."
			);
		}

		if (e <= 0) {
			throw new IllegalArgumentException(
				"\nEpsilon (" + eps + ") inválido."
			);
		}
		
		this.tA  = lr;
		this.eps = e;
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> AdaGrad </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA valor de taxa de aprendizado.
	 */
	public AdaGrad(Number tA) {
		this(tA, PADRAO_EPS);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> AdaGrad </strong>.
	 * <p>
	 *    Os hiperparâmetros do AdaGrad serão inicializados com os valores padrão.
	 * </p>
	 */
	public AdaGrad() {
		this(PADRAO_TA, PADRAO_EPS);
	}

	@Override
	public void construir(Modelo modelo) {
		initParams(modelo);

		ac = new Tensor[0];
		double valorInicial = 0.1;
		for (Tensor param : _params) {
			Tensor t = new Tensor(param.shape()).preencher(valorInicial);
			ac = utils.addEmArray(ac, t);
		}
		
		_construido = true;// otimizador pode ser usado
	}

	@Override
	public void atualizar() {
		verificarConstrucao();
		
		for (int i = 0; i < _params.length; i++) {
			ac[i].aplicar(ac[i], _grads[i], 
				(ac, g) -> ac += (g*g)
			);

			_params[i].aplicar(_params[i], _grads[i], ac[i], 
				(p, g, ac) -> p -= ((g * tA) / (Math.sqrt(ac) + eps))
			);
		}
	}

	@Override
	public String info() {
		verificarConstrucao();
		construirInfo();
		
		addInfo("Lr: " + tA);
		addInfo("Epsilon: " + eps);

		return super.info();
	}
}
