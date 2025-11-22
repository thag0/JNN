package jnn.inicializadores;

import java.util.Random;

import jnn.core.tensor.Tensor;

/**
 * Classe responsável pelas funções de inicialização dos pesos
 * da Rede Neural.
 */
public abstract class Inicializador {

	/**
	 * Gerador de números pseudo aleatórios compartilhado
	 * para as classes filhas.
	 */
	protected Random random = new Random();

	/**
	 * Inicialização com seed aleatória
	 */
	protected Inicializador() {}

	/**
	 * Configura o início do gerador aleatório.
	 * @param seed nova seed de início.
	 */
	public void setSeed(Number seed) {
		if (seed == null) {
			throw new NullPointerException("\nseed == null.");
		}

		random.setSeed(seed.longValue());
	}

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
		
		}else if (shape.length == 2) { //2D
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
	 * @param tensor tensor desejado.
	 */
	public abstract void forward(Tensor tensor);

	/**
	 * Retorna o nome do inicializador.
	 * @return nome do inicializador.
	 */
	public String nome() {
		return getClass().getSimpleName();
	}
}
