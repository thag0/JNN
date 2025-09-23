package jnn.otimizadores;

import jnn.core.tensor.Tensor;

/**
 * <h2>
 *    Gradient Descent
 * </h2>
 * Classe que implementa o algoritmo de Descida do Gradiente para otimização de redes neurais.
 * Atualiza diretamente os pesos da rede com base no gradiente.
 * <p>
 *    O Gradiente descendente funciona usando a seguinte expressão:
 * </p>
 * <pre>
 *    v -= g * tA
 * </pre>
 * Onde:
 * <p>
 *    {@code v} - variável que será otimizadada.
 * </p>
 *    {@code g} - gradiente correspondente a variável que será otimizada.
 * </p>
 * <p>
 *    {@code tA} - taxa de aprendizado do otimizador.
 * </p>
 */
public class GD extends Otimizador {

	/**
	 * Valor de taxa de aprendizado do otimizador.
	 */
	private final double tA;

	/**
	 * Inicializa uma nova instância de otimizador da <strong> Descida do Gradiente </strong>
	 * usando os valores de hiperparâmetros fornecidos.
	 * @param tA taxa de aprendizado do otimizador.
	 */
	public GD(Number tA) {
		double lr = tA.doubleValue();

		if (lr <= 0) {
			throw new IllegalArgumentException(
				"\nTaxa de aprendizado (" + lr + ") inválida."
			);
		}

		this.tA = lr;
	}

	/**
	 * Inicializa uma nova instância de otimizador da <strong> Descida do Gradiente </strong>.
	 * <p>
	 *    Os hiperparâmetros do GD serão inicializados com os valores padrão.
	 * </p>
	 */
	public GD() {
		this(0.01);
	}

	@Override
	public void construir(Tensor[] params, Tensor[] grads) {
		initParams(params, grads);
		_construido = true;// otimizador pode ser usado
	}

	@Override
	public void atualizar() {
		verificarConstrucao();
		
		for (int i = 0; i < _params.length; i++) {
			_params[i].sub(_grads[i]);
		} 
	}

	@Override
	public String info() {
		verificarConstrucao();
		construirInfo();
		
		addInfo("Lr: " + tA);

		return super.info();
	}
	
}
