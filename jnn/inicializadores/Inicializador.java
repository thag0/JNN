package jnn.inicializadores;

import jnn.core.tensor.Tensor;

/**
 * Classe responsável pelas funções de inicialização dos pesos
 * da Rede Neural.
 */
public abstract class Inicializador {

	/**
	 * Inicialização com seed aleatória
	 */
	protected Inicializador() {}

	/**
	 * Calcula os valores de fanIn e fanOut para ser usado pelos inicializadores.
	 * @param tensor {@code Tensor} que será inicializado.
	 * @return array contendo {@code [fanIn, fanOut]}.
	 */
	protected int[] calcularFans(Tensor tensor) {
		int[] shape = tensor.shape();
		int[] fans = {0, 0};

		if (shape.length == 1) { //1D
			fans[0] = shape[0];
			fans[1] = shape[0];
		
		} else if (shape.length == 2) { //2D
			fans[0] = shape[0];
			fans[1] = shape[1];
		
		} else { // 3D+
			int fanIn = 1;
			for (int i = 0; i < shape.length - 1; i++) {
			  fanIn *= shape[i];
			}
		
			int fanOut = 1;
			for (int i = shape.length - 1; i >= 1; i--) {
			  fanOut *= shape[i];
			}
		
			fans[0] = fanIn;
			fans[1] = fanOut;
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
	public String nome() {
		return getClass().getSimpleName();
	}
}
