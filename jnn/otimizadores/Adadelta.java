package jnn.otimizadores;

import jnn.camadas.Camada;
import jnn.core.tensor.Variavel;

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
	private final double epsilon;

	/**
	 * Acumuladores para os kernels.
	 */
	private Variavel[] ac;

	/**
	 * Acumuladores para os bias.
	 */
	private Variavel[] acb;

	/**
	 * Acumulador atualizado para os kernels.
	 */
	private Variavel[] acAt;

	/**
	 * Acumulador atualizado para os bias.
	 */
	private Variavel[] acAtb;

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
		this.epsilon = eps;
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> Adadelta </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param rho valor de decaimento do otimizador.
	 * @param epsilon usado para evitar a divisão por zero.
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
	public void construir(Camada[] camadas) {
		int[] params = initParams(camadas);
		int kernels = params[0];
		int bias = params[1];

		ac    = initVars(kernels);
		acAt  = initVars(kernels);
		acb   = initVars(bias);
		acAtb = initVars(bias);

		_construido = true;// otimizador pode ser usado
	}

	@Override
	public void atualizar() {
		verificarConstrucao();
		
		int idKernel = 0, idBias = 0;

		for (Camada camada : _params) {
			Variavel[] kernel = camada.kernelParaArray();
			Variavel[] gradK = camada.gradKernelParaArray();
			idKernel = adadelta(kernel, gradK, ac, acAt, idKernel);

			if (camada.temBias()) {
				Variavel[] bias = camada.biasParaArray();
				Variavel[] gradB = camada.gradBiasParaArray();
				idBias = adadelta(bias, gradB, acb, acAtb, idBias);
			}
		}
	}

	/**
	 * Atualiza as variáveis usando o gradiente pré calculado.
	 * @param vars variáveis que serão atualizadas.
	 * @param grads gradientes das variáveis.
	 * @param ac acumulador do otimizador.
	 * @param acAt acumulador atualizado.
	 * @param id índice inicial das variáveis dentro do array de momentums.
	 * @return índice final após as atualizações.
	 */
	private int adadelta(Variavel[] vars, Variavel[] grads, Variavel[] ac, Variavel[] acAt, int id) {
		double g, delta;

		for (int i = 0; i < vars.length; i++) {
			g = grads[i].get();
			ac[id].set((rho * ac[id].get()) + ((1 - rho) * (g*g)));
			
			delta = Math.sqrt(acAt[id].get() + epsilon) / Math.sqrt(ac[id].get() + epsilon) * g;
			acAt[id].set((rho * acAt[id].get()) + ((1 - rho) * (delta * delta)));
			
			vars[i].sub(delta);
			
			id++;
		}

		return id;
	}

	@Override
	public String info() {
		verificarConstrucao();
		construirInfo();

		addInfo("Rho: " + rho);
		addInfo("Epsilon: " + epsilon);

		return super.info();
	}
}
