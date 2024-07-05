package jnn.otimizadores;

import jnn.camadas.Camada;
import jnn.core.tensor.Tensor;

/**
 * Implementação do algoritmo de otimização AMSGrad, que é uma variação do 
 * algoritmo Adam que resolve um problema de convergência em potencial do Adam.
 * <p>
 * 	Os hiperparâmetros do AMSGrad podem ser ajustados para controlar o 
 * 	comportamento do otimizador durante o treinamento.
 * </p>
 * <p>
 *    O AMSGrad funciona usando a seguinte expressão:
 * </p>
 * <pre>
 *    v -= (tA * mc) / ((√ vc) + eps)
 * </pre>
 * Onde:
 * <p>
 *    {@code p} - variável que será otimizada.
 * </p>
 * <p>
 *    {@code tA} - valor de taxa de aprendizado.
 * </p>
 * <p>
 *    {@code mc} - valor de momentum corrigido.
 * </p>
 * <p>
 *    {@code vc} - valor de momentum de segunda ordem corrigido.
 * </p>
 * Os valores de momentum corrigido (mc) e momentum de segunda ordem
 * corrigido (vc) se dão por:
 * <pre>
 *    mc = m / (1 - beta1ⁱ)
 * </pre>
 * <pre>
 *    vc = vC / (1 - beta2ⁱ)
 * </pre>
 * Onde:
 * <p>
 *    {@code m} - valor de momentum correspondete a variável que será otimizada.
 * </p>
 * <p>
 *    {@code vC} - valor de momentum de segunda ordem corrigido correspondente 
 * 	a variável que será otimizada.
 * </p>
 * <p>
 *    {@code i} - contador de interações do otimizador.
 * </p>
 * O valor de momentum de segunda ordem corrigido (vC) é dado por:
 * <pre>
 * vC = max(vC, v)
 * </pre>
 * Onde:
 * <p>
 *    {@code v} - coeficiente de momentum de segunda ordem correspondente a
 *		conexão do peso que está sendo atualizado.
 * </p>
 */
public class AMSGrad extends Otimizador {

    /**
	 * Valor padrão para a taxa de aprendizado do otimizador.
	 */
	private static final double PADRAO_TA = 0.001;

	/**
	 * Valor padrão para o decaimento do momento de primeira ordem.
	 */
	private static final double PADRAO_BETA1 = 0.95;
 
	/**
	 * Valor padrão para o decaimento do momento de segunda ordem.
	 */
	private static final double PADRAO_BETA2 = 0.999;
	 
	/**
	 * Valor padrão para epsilon.
	 */
	private static final double PADRAO_EPS = 1e-8;

    /**
	 * Valor de taxa de aprendizado do otimizador.
	 */
	private final double tA;

	/**
	 * Usado para evitar divisão por zero.
	 */
	private final double eps;

	/**
	 * Decaimento do momentum de primeira ordem.
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
	 * Coeficientes de momentum de segunda ordem.
	 */
	private Tensor[] v;

	/**
	 * Coeficientes de momentum de segunda ordem corrigidos.
	 */
	private Tensor[] vc;

	/**
	 * Contador de iterações.
	 */
	private long iteracoes;

	/**
	 * Inicializa uma nova instância de otimizador <strong> AMSGrad </strong> usando os 
	 * valores de hiperparâmetros fornecidos.
	 * @param tA valor de taxa de aprendizado.
	 * @param beta1 decaimento do momento.
	 * @param beta2 decaimento do momento de segunda ordem.
	 * @param eps usado para evitar a divisão por zero.
	 */
	public AMSGrad(double tA, double beta1, double beta2, double eps) {
		if (tA <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de aprendizado (" + tA + ") inválida."
			);
		}
		if (beta1 <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de decaimento de primeira ordem (" + beta1 + ") inválida."
			);
		}
		if (beta2 <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de decaimento de segunda ordem (" + beta2 + ") inválida."
			);
		}
		if (eps <= 0) {
			throw new IllegalArgumentException(
			"\nEpsilon (" + eps + ") inválido."
			);
		}
		
		this.tA = tA;
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.eps = eps;
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> AMSGrad </strong> usando os 
	 * valores de hiperparâmetros fornecidos.
	 * @param tA valor de taxa de aprendizado.
	 * @param beta1 decaimento do momento.
	 * @param beta2 decaimento do momento de segunda ordem.
	 */
	public AMSGrad(double tA, double beta1, double beta2) {
	  this(tA, beta1, beta2, PADRAO_EPS);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> AMSGrad </strong> usando os 
	 * valores de hiperparâmetros fornecidos.
	 * @param tA valor de taxa de aprendizado.
	 */
	public AMSGrad(double tA){
		this(tA, PADRAO_BETA1, PADRAO_BETA2, PADRAO_EPS);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> AMSGrad </strong>.
	 * <p>
	 *		Os hiperparâmetros do AMSGrad serão inicializados com os valores padrão.
	 * </p>
	 */
	public AMSGrad(){
		this(PADRAO_TA, PADRAO_BETA1, PADRAO_BETA2, PADRAO_EPS);
	}

	@Override
	public void construir(Camada[] camadas) {
		initParams(camadas);

		m  = new Tensor[0];
		v  = new Tensor[0];
		vc = new Tensor[0];
		for (Tensor t : _params) {
			m  = utils.addEmArray(m,  new Tensor(t.shape()));
			v  = utils.addEmArray(v,  new Tensor(t.shape()));
			vc = utils.addEmArray(vc, new Tensor(t.shape()));
		}

		_construido = true;// otimizador pode ser usado
	}

	@Override
	public void atualizar() {
		verificarConstrucao();

		iteracoes++;
		double fb1 = (1 - Math.pow(beta1, iteracoes));
		double fb2 = (1 - Math.pow(beta2, iteracoes));
		
		for (int i = 0; i < _params.length; i++) {
			m[i].aplicar(m[i], _grads[i], (m, g) ->
				(beta1 * m) + ((1 - beta1) * g)
			);
			v[i].aplicar(v[i], _grads[i], 
				(v, g) -> (beta2 * v) + ((1 - beta2) * (g*g))
			);
			vc[i].aplicar(vc[i], v[i],
				(vc, v) -> Math.max(vc, v)
			);
			_params[i].aplicar(_params[i], m[i], v[i],
				(p, m, v) -> p -= ((m/fb1) * tA) / (Math.sqrt(v/fb2) + eps)
			);
		}
	}

	@Override
	public String info() {
		verificarConstrucao();
		construirInfo();
	  
		addInfo("Lr: " + tA);
		addInfo("Beta1: " + beta1);
		addInfo("Beta2: " + beta2);
		addInfo("Epsilon: " + eps);

		return super.info();
	}

}
