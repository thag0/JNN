package jnn.otimizadores;

import jnn.camadas.Camada;
import jnn.core.tensor.Variavel;

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
	private final double epsilon;

	/**
	 * Fator de decaimento.
	 */
	private final double rho;

	/**
	 * Acumuladores para os kernels
	 */
	private Variavel[] ac;

	/**
	 * Acumuladores para os bias.
	 */
	private Variavel[] acb;

	/**
	 * Inicializa uma nova instância de otimizador <strong> RMSProp </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA valor de taxa de aprendizagem.
	 * @param rho fator de decaimento do RMSProp.
	 * @param eps usado para evitar a divisão por zero.
	 */
	public RMSProp(double tA, double rho, double eps) {
		if (tA <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de aprendizagem (" + tA + "), inválida."
			);
		}
		if (rho <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de decaimento (" + rho + "), inválida."
			);
		}
		if (eps <= 0) {
			throw new IllegalArgumentException(
				"\nEpsilon (" + eps + "), inválido."
			);
		}

		this.tA = tA;
		this.rho = rho;
		this.epsilon = eps;
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> RMSProp </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA valor de taxa de aprendizagem.
	 * @param rho fator de decaimento do RMSProp.
	 */
	public RMSProp(double tA, double rho) {
		this(tA, rho, PADRAO_EPS);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> RMSProp </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA valor de taxa de aprendizagem.
	 */
	public RMSProp(double tA) {
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
	public void construir(Camada[] camadas) {
		int[] params = initParams(camadas);
		int kernels = params[0];
		int bias = params[1];

		this.ac  = initVars(kernels);
		this.acb = initVars(bias);
		
		_construido = true;// otimizador pode ser usado
	}

	@Override
	public void atualizar() {
		verificarConstrucao();
		
		int idKernel = 0, idBias = 0;
		for (Camada camada : _params) {
			Variavel[] kernel = camada.kernelParaArray();
			Variavel[] gradK = camada.gradKernelParaArray();
			idKernel = rmsprop(kernel, gradK, ac, idKernel);

			if (camada.temBias()) {
				Variavel[] bias = camada.biasParaArray();
				Variavel[] gradB = camada.gradBiasParaArray();
				idBias = rmsprop(bias, gradB, acb, idBias);
			}
		}
	}

	/**
	 * Atualiza as variáveis usando o gradiente pré calculado.
	 * @param vars variáveis que serão atualizadas.
	 * @param grads gradientes das variáveis.
	 * @param ac acumulador do otimizador.
	 * @param id índice inicial das variáveis dentro do array de momentums.
	 * @return índice final após as atualizações.
	 */
    private int rmsprop(Variavel[] vars, Variavel[] grads, Variavel[] ac, int id) {
        double g;

        for (int i = 0; i < vars.length; i++) {
            g = grads[i].get();
            
			ac[id].set(rho * ac[id].get() + (1 - rho) * g * g);
            vars[i].sub((g * tA) / (Math.sqrt(ac[id].get()) + epsilon));

            id++;
        }

        return id;
    }

	@Override
	public String info() {
		verificarConstrucao();
		construirInfo();
		
		addInfo("Lr: " + tA);
		addInfo("Rho: " + rho);
		addInfo("Epsilon: " + epsilon);

		return super.info();
	}

}
