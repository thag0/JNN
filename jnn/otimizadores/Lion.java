package jnn.otimizadores;

import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;

/**
 * <h2>
 *		Lion
 * </h2>
 *		O otimizador Lion é uma alternativa mais eficientes em
 *		memória que utiliza apenas o sinal de operação para controlar
 *		a magnitude das suas atualizações.
 * <p>
 *    O Lion funciona usando a seguinte expressão:
 * </p>
 * <pre>
 *v -= tA * signum((m * beta1) + (g * (1 - beta2)))
 *m -= (m * beta2) + (g (1 - beta2))
 *</pre>
 * Onde:
 * <p>
 *    {@code v} - variável que será otimizada.
 * </p>
 * <p>
 *    {@code tA} - taxa de aprendizado de otimizador.
 * </p>
 * <p>
 *    {@code m} - coeficientes de momentum do otimizador.
 * </p>
 * <p>
 *    {@code g} - gradiente em relação a variável que será otimizada.
 * </p>
 * <p>
 *    {@code beta1 e beta2} - valores de decaimento dos momentums de primeira
 *    e segunda ordem.
 * </p>
 */
public class Lion extends Otimizador {

	/**
	 * Valor de taxa de aprendizado padrão do otimizador.
	 */
	private static final double PADRAO_TA = 0.001;

	/**
	 * Valor padrão para o decaimento do momento de primeira ordem.
	 */
	private static final double PADRAO_BETA1 = 0.9;
 
	/**
	 * Valor padrão para o decaimento do momento de segunda ordem.
	 */
	private static final double PADRAO_BETA2 = 0.99;

	/**
	 * Valor de taxa de aprendizado do otimizador.
	 */
	private final double tA;

	/**
	 * Decaimento do momentum.
	 */
	private final double beta1;
	 
	/**
	 * Decaimento do momentum de segunda ordem.
	 */
	private final double beta2;

	/**
	 * Coeficientes de momentum.
	 */
    private Tensor[] m;

	/**
	 * Inicializa uma nova instância de otimizador <strong> Lion </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA taxa de aprendizado do otimizador.
	 * @param beta1 taxa de decaimento de primeira ordem.
	 * @param beta2 taxa de decaimento de segunda ordem.
	 */
    public Lion(Number tA, Number beta1, Number beta2) {
		double lr = tA.doubleValue();
		double b1 = beta1.doubleValue();
		double b2 = beta2.doubleValue();

		if (lr <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de aprendizado (" + lr + ") inválida."
			);
		}
		if (b1 <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de decaimento de primeira ordem (" + b1 + ") inválida."
			);
		}
		if (b2 <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de decaimento de segunda ordem (" + b2 + ") inválida."
			);
		}
		
		this.tA 	 = lr;
		this.beta1 	 = b1;
		this.beta2 	 = b2;
    }

	/**
	 * Inicializa uma nova instância de otimizador <strong> Lion </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA taxa de aprendizado do otimizador
	 */
	public Lion(Number tA) {
		this(tA, PADRAO_BETA1, PADRAO_BETA2);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> Lion </strong>.
	 * <p>
	 *    Os hiperparâmetros do Lion serão inicializados com seus os valores padrão.
	 * </p>
	 */
    public Lion() {
        this(PADRAO_TA, PADRAO_BETA1, PADRAO_BETA2);
    }

    @Override
    public void construir(Modelo modelo) {
        initParams(modelo);

        m = new Tensor[0];
        for (Tensor param : _params) {
            m = utils.addEmArray(m, new Tensor(param.shape()));
        }
    }

    @Override
    public void atualizar() {
        for (int i = 0; i < _params.length; i++) {
            _params[i].aplicar(_params[i], _grads[i], m[i],
                (p, g, m) -> p -= tA * Math.signum((m * beta1) + (g * (1.0 - beta1)))
            );

            m[i].aplicar(m[i], _grads[i],
                (m, g) -> m -= (m * beta2) + (g * (1.0 - beta2))
            );
        }
    }

    @Override
    public String info() {
        construirInfo();

        addInfo("Lr: " + tA);
        addInfo("Beta1: " + beta1);
        addInfo("Beta2: " + beta2);

        return super.info();
    }
    
}
