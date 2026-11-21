package jnn.initializers;

import jnn.core.tensor.Tensor;

/**
 * Classe responsável pelas funções de inicialização dos pesos
 * da Rede Neural.
 */
public abstract class Initializer {

	/**
	 * Inicialização com seed aleatória
	 */
	protected Initializer() {}

	/**
	 * Calcula os valores de fanIn e fanOut para ser usado pelos inicializadores.
	 * @param tensor {@code Tensor} que será inicializado.
	 * @return array contendo {@code [fanIn, fanOut]}.
	 */
	protected int[] calculateFans(Tensor tensor) {
		int[] shape = tensor.shape();
		int[] fans = {0, 0};

		if (shape.length == 1) { //1D
			fans[0] = shape[0];
			fans[1] = shape[0];
		
		} else if (shape.length == 2) { //2D
			fans[0] = shape[0];
			fans[1] = shape[1];
		
		} else { // 3D+ (no caso para layout NCHW)
			int campoReceptivo = 1;
			for (int i = 2; i < shape.length; i++) {
				campoReceptivo *= shape[i];
			}

			fans[0] = shape[0] * campoReceptivo;
			fans[1] = shape[1] * campoReceptivo;
		}

		return fans;
	}

	/**
	 * Inicializa todos os valores tensor.
	 * @param t tensor desejado.
	 */
	public abstract void forward(Tensor t);

	/**
	 * Retorna o nome do inicializador.
	 * @return nome do inicializador.
	 */
	public String name() {
		return getClass().getSimpleName();
	}
}
