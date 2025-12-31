package jnn.otm;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;
import jnn.core.tensor.TensorData;

/**
 * <h2>
 *		Lion
 * </h2>
 * <p>
 *		O otimizador Lion é uma alternativa mais eficientes em
 *		memória que utiliza apenas o sinal de operação para controlar
 *		a magnitude das suas atualizações.
 * </p>
 * {@link {@code Paper} http://arxiv.org/abs/2302.06675}
 */
public class Lion extends Otimizador {

	/**
	 * Valor de taxa de aprendizado padrão do otimizador.
	 */
	private static final double PADRAO_LR = 0.001;

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
	private final double lr;

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
    private Tensor[] m = {};

	/**
	 * Inicializa uma nova instância de otimizador <strong> Lion </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param lr taxa de aprendizado do otimizador.
	 * @param beta1 taxa de decaimento de primeira ordem.
	 * @param beta2 taxa de decaimento de segunda ordem.
	 */
    public Lion(Number lr, Number beta1, Number beta2) {
		double lr_ = lr.doubleValue();
		double b1_ = beta1.doubleValue();
		double b2_ = beta2.doubleValue();

		if (lr_ <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de aprendizado (" + lr_ + ") inválida."
			);
		}
		if (b1_ <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de decaimento de primeira ordem (" + b1_ + ") inválida."
			);
		}
		if (b2_ <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de decaimento de segunda ordem (" + b2_ + ") inválida."
			);
		}
		
		this.lr 	 = lr_;
		this.beta1 	 = b1_;
		this.beta2 	 = b2_;
    }

	/**
	 * Inicializa uma nova instância de otimizador <strong> Lion </strong> 
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param lr taxa de aprendizado do otimizador
	 */
	public Lion(Number lr) {
		this(lr, PADRAO_BETA1, PADRAO_BETA2);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> Lion </strong>.
	 * <p>
	 *    Os hiperparâmetros do Lion serão inicializados com seus os valores padrão.
	 * </p>
	 */
    public Lion() {
        this(PADRAO_LR, PADRAO_BETA1, PADRAO_BETA2);
    }

    @Override
	public void construir(Tensor[] params, Tensor[] grads) {
		initParams(params, grads);
		
        for (Tensor param : _params) {
            m = JNNutils.addEmArray(m, new Tensor(param.shape()));
        }

		_construido = true;// otimizador pode ser usado
    }

    @Override
    public void update() {
		checkInicial();

		final int n = _params.length;
        for (int i = 0; i < n; i++) {
			TensorData p_i = _params[i].data();
			TensorData g_i = _grads[i].data();
			TensorData m_i = m[i].data();

			// p -= tA * signum(β1*m + (1-β1)*g)
			TensorData temp = m_i.clone().mul(beta1).add(g_i, 1.0 - beta1);
			p_i.add(temp.signum().mul(-lr));

			// m = (β2 * m) + ((1-β2) * g)
			m_i.mul(beta2).add(g_i, 1.0 - beta2);
        }
    }

    @Override
    public String info() {
        construirInfo();

        addInfo("Lr: " + lr);
        addInfo("Beta1: " + beta1);
        addInfo("Beta2: " + beta2);

        return super.info();
    }
    
}
