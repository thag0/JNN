package jnn.camadas;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

/**
 * <h2>
 *    Camada de entrada
 * </h2>
 * <p>
 *    A camada de entrada serve apenas para facilitar a criação de modelos 
 *    e não tem impacto após a compilação deles.
 * </p>
 * <p>
 *    Ela deve estar no início das camadas de um modelo para ser considerada,
 *    e no momento da compilação é destruída e servirá de base para a primeira 
 *    camada do modelo ser construída.
 * </p>
 * Exemplo
 * <pre>
 * Sequencial modelo = new Sequencial(
 *    Entrada(28, 28),
 *    Flatten(),
 *    Densa(20, "sigmoid"),
 *    Densa(20, "sigmoid"),
 *    Densa(10, "softmax")
 * );
 * </pre>
 */
public class Entrada extends Camada {

	/**
	 * Formato usado para entrada de um modelo.
	 */
	private int[] shape;

	/**
	 * Inicializa um camada de entrada de acordo com o formato especificado.
	 * @param shape formato de entrada usado para o modelo em que a camada estiver.
	 */
	public Entrada(int... shape) {
		JNNutils.validarNaoNulo(shape, "Formato recebido é nulo.");

		if (shape.length < 1) {
			throw new UnsupportedOperationException(
				"\nO formato recebido deve conter ao menos um elemento."
			);
		}

		if (!JNNutils.apenasMaiorZero(shape)) {
			throw new IllegalArgumentException(
				"\nOs elementos do formato devem ser maiores que zero."
			);
		}

		this.shape = shape.clone();
	}

	@Override
	public void construir(int[] shape) {}

	@Override
	public void inicializar() {}

	@Override
	public Tensor forward(Tensor x) {
		return x;
	}

	@Override
	public Tensor backward(Tensor g) {
		return g;
	}

	@Override
	public Tensor saida() {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui retorno de saída."
		);
	}

	@Override
	public int[] shapeIn() {
		return shape;
	}

	@Override
	public int[] shapeOut() {
		return shapeIn();
	}

	@Override
	public int numParams() {
		return 0;
	}
	
}
