package jnn.otimizadores;

import jnn.camadas.Camada;
import jnn.core.tensor.Variavel;

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
 *    será otimizada.d
 * </p>
 * <p>
 *    {@code eps} - um valor pequeno para evitar divizões por zero.
 * </p>
 */
public class AdaGrad extends Otimizador {

	/**
	 * Valor padrão para a taxa de aprendizagem do otimizador.
	 */
	private static final double PADRAO_TA = 0.5;

	/**
	 * Valor padrão para o valor de epsilon pro otimizador.
	 */
	private static final double PADRAO_EPS = 1e-7; 

	/**
	 * Valor de taxa de aprendizagem do otimizador.
	 */
	private double taxaAprendizagem;

	/**
	 * Usado para evitar divisão por zero.
	 */
	private double epsilon;

	/**
	 * Acumuladores para os kernels.
	 */
	private Variavel[] ac;

	/**
	 * Acumuladores para os bias.
	 */
	private Variavel[] acb;

	/**
	 * Inicializa uma nova instância de otimizador <strong> AdaGrad </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA valor de taxa de aprendizagem.
	 * @param eps usado para evitar a divisão por zero.
	 */
	public AdaGrad(double tA, double eps) {
		if (tA <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de aprendizagem (" + tA + "), inválida."
			);
		}

		if (eps <= 0) {
			throw new IllegalArgumentException(
				"\nEpsilon (" + eps + "), inválido."
			);
		}
		
		this.taxaAprendizagem = tA;
		this.epsilon = eps;
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> AdaGrad </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA valor de taxa de aprendizagem.
	 */
	public AdaGrad(double tA) {
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
	public void construir(Camada[] camadas) {
		int nKernel = 0;
		int nBias = 0;
		
		for (Camada camada : camadas) {
			if (!camada.treinavel()) continue;

			nKernel += camada.kernel().tamanho();
			if (camada.temBias()) nBias += camada.bias().tamanho();
		}

		this.ac  = initVars(nKernel);
		this.acb = initVars(nBias);
		
		double valorInicial = 0.1;
		for (int i = 0; i < nKernel; i++) {
			ac[i].set(valorInicial);
		}
		for (int i = 0; i < nBias; i++) {
			acb[i].set(valorInicial);
		}
		
		_construido = true;//otimizador pode ser usado
	}

	@Override
	public void atualizar(Camada[] camadas) {
		verificarConstrucao();
		
		int idKernel = 0, idBias = 0;
		for (Camada camada : camadas) {
			if (!camada.treinavel()) continue;

			Variavel[] kernel = camada.kernelParaArray();
			Variavel[] gradK = camada.gradKernelParaArray();
			idKernel = calcular(kernel, gradK, ac, idKernel);
			camada.setKernel(kernel);
			
			if (camada.temBias()) {
				Variavel[] bias = camada.biasParaArray();
				Variavel[] gradB = camada.gradBiasParaArray();
				idBias = calcular(bias, gradB, acb, idBias);
				camada.setBias(bias);
			}
		}
	}

	/**
	 * Atualiza as variáveis usando o gradiente pré calculado.
	 * @param vars variáveis que serão atualizadas.
	 * @param grads gradientes das variáveis.
	 * @param acumulador acumulador do otimizador.
	 * @param id índice inicial das variáveis dentro do array de momentums.
	 * @return índice final após as atualizações.
	 */
	private int calcular(Variavel[] vars, Variavel[] grads, Variavel[] acumulador, int id) {
		double g;
		for (int i = 0; i < vars.length; i++) {
			g = grads[i].get();
			acumulador[id].add(g*g);
			vars[i].sub((g * taxaAprendizagem) / (Math.sqrt(ac[id].get() + epsilon)));
			id++;
		}

		return id;
	}

	@Override
	public String info() {
		verificarConstrucao();
		construirInfo();
		
		addInfo("TaxaAprendizagem: " + taxaAprendizagem);
		addInfo("Epsilon: " + epsilon);

		return super.info();
	}
}
