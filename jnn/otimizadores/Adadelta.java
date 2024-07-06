package jnn.otimizadores;

import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;

/**
 * Implementação do otimizador Adadelta.
 * <p>
 * 	Os hiperparâmetros do Adadelta podem ser ajustados para controlar o 
 * 	comportamento do otimizador durante o treinamento.
 * </p>
 * <p>
 *    O Adadelta funciona usando a seguinte expressão:
 * </p>
 * <pre>
 *    v -= delta
 * </pre>
 * Onde delta é dado por:
 * <pre>
 * delta = √(acAt + eps) / √(ac + eps) * g
 * </pre>
 * Onde:
 * <p>
 *    {@code v} - variável que será otimizada.
 * </p>
 * <p>
 *    {@code acAt} - acumulador atualizado correspondente a variável que
 *    será otimizada.
 * </p>
 * <p>
 *    {@code ac} - acumulador correspondente a variável que
 *    será otimizada
 * </p>
 * <p>
 *    {@code g} - gradientes correspondente a variável que será otimizada.
 * </p>
 * Os valores do acumulador (ac) e acumulador atualizado (acAt) se dão por:
 * <pre>
 *ac   = (rho * ac)   + ((1 - rho) * g²)
 *acAt = (rho * acAt) + ((1 - rho) * delta²)
 * </pre>
 * Onde:
 * <p>
 *    {@code rho} - constante de decaimento do otimizador.
 * </p>
 */
public class Adadelta extends Otimizador {

	/**
	 * Valor padrão para a taxa de decaimento.
	 */
	private static final double PADRAO_RHO = 0.999;

	/**
	 * Valor padrão para epsilon.
	 */
	private static final double PADRAO_EPS = 1e-6;

	/**
	 * Constante de decaimento do otimizador.
	 */
	private final double rho;

	/**
	 * Valor usado para evitar divisão por zero.
	 */
	private final double eps;

	/**
	 * Acumuladores.
	 */
	private Tensor[] ac;

	/**
	 * Deltas de atualização
	 */
	private Tensor[] deltas;

	/**
	 * Acumuladores atualizados.
	 */
	private Tensor[] acAt;

	/**
	 * Inicializa uma nova instância de otimizador <strong> Adadelta </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param rho valor de decaimento do otimizador.
	 * @param eps pequeno valor usado para evitar a divisão por zero.
	 */
	public Adadelta(double rho, double eps) {
		if (rho <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de decaimento (" + rho + ") inválida."
			);
		}

		if (eps <= 0) {
			throw new IllegalArgumentException(
				"\nEpsilon (" + eps + ") inválido."
			);
		}

		this.rho = rho;
		this.eps = eps;
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> Adadelta </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param rho valor de decaimento do otimizador.
	 * @param eps usado para evitar a divisão por zero.
	 */
	public Adadelta(double rho) {
		this(rho, PADRAO_EPS);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> Adadelta </strong>.
	 * <p>
	 *    Os hiperparâmetros do Adadelta serão inicializados com os valores padrão.
	 * </p>
	 */
	public Adadelta() {
		this(PADRAO_RHO, PADRAO_EPS);
	}

	@Override
	public void construir(Modelo modelo) {
		initParams(modelo);

		ac     = new Tensor[0];
		deltas = new Tensor[0];
		acAt   = new Tensor[0];
		for (Tensor t : _params) {
			ac     = utils.addEmArray(ac,     new Tensor(t.shape()));
			deltas = utils.addEmArray(deltas, new Tensor(t.shape()));
			acAt   = utils.addEmArray(acAt,   new Tensor(t.shape()));
		}

		_construido = true;// otimizador pode ser usado
	}

	@Override
	public void atualizar() {
		verificarConstrucao();
		
		for (int i = 0; i < _params.length; i++) {
			ac[i].aplicar(ac[i], _grads[i],
				(ac, g) -> (rho * ac) + ((1 - rho) * (g*g))
			);

			deltas[i].aplicar(acAt[i], ac[i], _grads[i], 
				(acat, ac, g) -> Math.sqrt(acat + eps) / Math.sqrt(ac + eps) * g
			);

			acAt[i].aplicar(acAt[i], deltas[i], 
				(acat, d) -> (rho * acat) + ((1 - rho) * (d*d))
			);

			_params[i].sub(deltas[i]);
		}
	}

	@Override
	public String info() {
		verificarConstrucao();
		construirInfo();

		addInfo("Rho: " + rho);
		addInfo("Epsilon: " + eps);

		return super.info();
	}
}
