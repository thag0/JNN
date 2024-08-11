package jnn.camadas;

import jnn.core.Utils;
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
	 * Utilitário.
	 */
	private Utils utils = new Utils();

	/**
	 * Formato usado para entrada de um modelo.
	 */
	private int[] shape;

	/**
	 * Inicializa um camada de entrada de acordo com o formato especificado.
	 * </pre>
	 * @param shape formato de entrada usado para o modelo em que a camada estiver.
	 */
	public Entrada(int... shape) {
		utils.validarNaoNulo(shape, "Formato recebido é nulo.");

		if (shape.length < 1) {
			throw new UnsupportedOperationException(
				"\nO formato recebido deve conter ao menos um elemento."
			);
		}

		if (!utils.apenasMaiorZero(shape)) {
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
	public Tensor forward(Object x) {
		if (x instanceof Tensor) {
			return (Tensor) x;
		
		} else {
			throw new IllegalArgumentException(
				"\nTipo de entrada \"" + x.getClass().getTypeName() + "\"" +
				" não suportada."
			);
		}
	}

	@Override
	public Tensor backward(Object grad) {
		if (grad instanceof Tensor) {
			return (Tensor) grad;
		
		} else {
			throw new IllegalArgumentException(
				"\nTipo de gradiente \"" + grad.getClass().getTypeName() + "\"" +
				" não suportada."
			);
		}
	}

	@Override
	public Tensor saida() {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui retorno de saída."
		);
	}

	@Override
	public int[] shapeEntrada() {
		return shape;
	}

	@Override
	public int[] shapeSaida() {
		return shapeEntrada();
	}

	@Override
	public int numParams() {
		return 0;
	}
	
	@Override
	public void copiarParaTreinoLote(Camada camada) {
		throw new UnsupportedOperationException(
			"\nNão suportado."
		);
	}
}
