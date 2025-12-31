package jnn.otm;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;
import jnn.core.tensor.TensorData;

/**
 * <h2>
 *    Adaptive Delta
 * </h2>
 * Implementação do algoritmo de otimização Adadelta.
 * <p>
 *		O algoritmo ajusta os pesos do modelo usando gradiente descendente adaptativo,
 *		acumulando a média móvel dos quadrados dos gradientes e dos incrementos dos 
 *		parâmetros, eliminando a necessidade de uma taxa de aprendizado fixa.
 * </p>
 * {@link {@code Paper}: http://arxiv.org/abs/1212.5701}
 */
public class Adadelta extends Otimizador {

	/**
	 * Valor padrão para a taxa de decaimento.
	 */
	private static final double PADRAO_RHO = 0.99;

	/**
	 * Valor padrão para epsilon.
	 */
	private static final double PADRAO_EPS = 1e-8;

	/**
	 * Constante de decaimento do otimizador.
	 */
	private final double rho;

	/**
	 * Valor usado para evitar divisão por zero.
	 */
	private final double eps;

	/**
	 * Acumuladores dos gradientes ao quadrado.
	 */
	private Tensor[] acg = {};

	/**
	 * Acumuladores dos deltas ao quadrado.
	 */
	private Tensor[] acd = {};

	/**
	 * Deltas de atualização
	 */
	private Tensor[] deltas = {};

	/**
	 * Inicializa uma nova instância de otimizador <strong> Adadelta </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param rho valor de decaimento do otimizador.
	 * @param eps pequeno valor usado para evitar a divisão por zero.
	 */
	public Adadelta(Number rho, Number eps) {
		double r = rho.doubleValue();
		double e = eps.doubleValue();

		if (r <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de decaimento (" + r + ") inválida."
			);
		}

		if (e <= 0) {
			throw new IllegalArgumentException(
				"\nEpsilon (" + e + ") inválido."
			);
		}

		this.rho = r;
		this.eps = e;
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> Adadelta </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param rho valor de decaimento do otimizador.
	 * @param eps usado para evitar a divisão por zero.
	 */
	public Adadelta(Number rho) {
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
	public void construir(Tensor[] params, Tensor[] grads) {
		initParams(params, grads);

		for (Tensor param : _params) {
			acg     = JNNutils.addEmArray(acg,     new Tensor(param.shape()));
			deltas = JNNutils.addEmArray(deltas, new Tensor(param.shape()));
			acd   = JNNutils.addEmArray(acd,   new Tensor(param.shape()));
		}

		_construido = true;// otimizador pode ser usado
	}

	@Override
	public void update() {
		checkInicial();
		
		final int n = _params.length;
		for (int i = 0; i < n; i++) {
			TensorData p_i   = _params[i].data();
			TensorData g_i   = _grads[i].data();
			TensorData acg_i = acg[i].data();// E[g²]
			TensorData acd_i = acd[i].data();// E[Δx²]
			TensorData d_i   = deltas[i].data();

			// E[g²] = (rho * E[g²]) + ((1 - rho) * g²)
			acg_i.mul(rho).addcmul(g_i, g_i, 1.0 - rho);

			// delta = (sqrt(E[Δx²] + eps) / sqrt(E[g²] + eps))
			TensorData den = acg_i.clone().add(eps).sqrt();
			TensorData num = acd_i.clone().add(eps).sqrt();

			// delta = - (num/den) * g
			d_i.copiar(num).mul(-1).div(den).mul(g_i);

			// E[Δx²] = (rho * E[Δx²]) + ((1 - rho) * (delta²))
			acd_i.mul(rho).addcmul(d_i, d_i, 1.0 - rho);
			
			// p += delta
			p_i.add(d_i);
		}
	}

	@Override
	public String info() {
		checkInicial();
		construirInfo();

		addInfo("Rho: " + rho);
		addInfo("Epsilon: " + eps);

		return super.info();
	}
}
