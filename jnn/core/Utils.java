package jnn.core;

import java.lang.reflect.Array;

import jnn.core.tensor.Tensor;

/**
 * Utilitário geral para a biblioteca.
 */
public class Utils {

	/**
	 * Utilitário geral para a biblioteca.
	 */
	public Utils() {}

	/**
	 * Retorna o último índice válido do array.
	 * @param arr array base.
	 * @return último índice.
	 */
	public int ultimoIndice(Object arr) {
		if (arr instanceof int[]) {
			int[] a = (int[]) arr;

			if (a.length == 0) {
				throw new IllegalArgumentException("\nArray de tamanho 0.");
			}

			return a.length-1;
		} else {
			throw new IllegalArgumentException(
				"Tipo de dado (" + arr.getClass().getTypeName() + ") não suportado."
			);
		}
	}

	/**
	 * Verifica se o conteúdo do array contém valores maiores que zero.
	 * @param arr array base.
	 * @return resultado da verificação.
	 */
	public boolean apenasMaiorZero(int[] arr) {
		for (int i = 0; i < arr.length; i++) {
			if (arr[i] < 1) return false;
		}

		return true;
	}

	/**
	 * Formata as dimensões do array,
	 * <p>
	 *    Exemplo:
	 * </p>
	 * <pre>
	 *int[] arr = {2, 3};
	 *arr.shapeStr == "(2, 3)"
	 * </pre>
	 * @param arr array desejado.
	 * @return formato das dimensões do array
	 */
	public String shapeStr(int[] arr) {
		StringBuilder sb = new StringBuilder();

		sb.append("(");

		sb.append(arr[0]);
		for (int i = 1; i < arr.length; i++) {
			sb.append(", " + arr[i]);
		}

		sb.append(")");

		return sb.toString();
	}

	/**
	 * Converte um array primitivo para um array de tensores, cada tensor
	 * contendo um único valor do array.
	 * @param array array desejado.
	 * @return array de {@code Tensores}.
	 */
	public Tensor[] array1DParaTensors(double[] array) {
		int n = array.length;

		Tensor[] arr = new Tensor[n];
		for (int i = 0; i < n; i++) {
			arr[i] = new Tensor(new double[]{ array[i] }, 1);
			arr[i].nome("amostra " + i);// ajudar no debug
		}

		return arr;
	}

	/**
	 * Converte um array primitivo para um array de tensores, cada tensor contendo
	 * um {@code array1D}.
	 * @param array array desejado.
	 * @return array de {@code Tensores}.
	 */
	public Tensor[] array2DParaTensors(double[][] array) {
		int lin = array.length;
		int col = array[0].length;

		Tensor[] arr = new Tensor[lin];
		for (int i = 0; i < lin; i++) {
			arr[i] = new Tensor(col);
			arr[i].copiarElementos(array[i]);
			arr[i].nome("amostra " + i);// ajudar no debug
		}

		return arr;
	}

	/**
	 * Converte um array primitivo para um array de tensores, cada tensor contendo
	 * um {@code array2D}.
	 * @param array array desejado.
	 * @return array de {@code Tensores}.
	 */
	public Tensor[] array3DParaTensors(double[][][] array) {
		Tensor[] arr = new Tensor[array.length];

		int n = array.length;
		for (int i = 0; i < n; i++) {
			arr[i] = new Tensor(array[i]);
			arr[i].nome("amostra " + i);// ajudar no debug
		}

		return arr;	
	}

	/**
	 * Converte um array primitivo para um array de tensores, cada tensor contendo
	 * um {@code array3D}.
	 * @param array array desejado.
	 * @return array de {@code Tensores}.
	 */
	public Tensor[] array4DParaTensors(double[][][][] array) {
		Tensor[] arr = new Tensor[array.length];

		int n = array.length;
		for (int i = 0; i < n; i++) {
			arr[i] = new Tensor(array[i]);
			arr[i].nome("amostra " + i);// ajudar no debug
		}

		return arr;
	}

	/**
	 * Verifica se o objeto recebido é nulo.
	 * @param obj objeto de comparação.
	 * @param msg mensagem personalizada de erro.
	 * @throws NullPointerException caso o objeto seja nulo.
	 */
	public void validarNaoNulo(Object obj, String msg) {
		if (obj == null) {
			String str;

			if (msg == null) {
				str = "Objeto recebido é nulo.";
				
			} else {
				msg = msg.trim();
				str = msg.isEmpty() ? "Objeto recebido é nulo." : msg;
			}

			throw new NullPointerException("\n" + str);
		}
	}

	/**
	 * Adiciona um novo elemento ao array.
	 * @param <T> tipo dos elementos do array
	 * @param arr {@code array}.
	 * @param elm elemento para adição.
	 * @return novo array com elemento adicionado.
	 */
	public <T> T[] addEmArray(T[] arr, T elm) {
		validarNaoNulo(arr, "Array nulo");
		validarNaoNulo(elm, "elemento nulo");
		
		@SuppressWarnings("unchecked")
		T[] novo = (T[]) Array.newInstance(arr.getClass().getComponentType(), arr.length + 1);

		System.arraycopy(arr, 0, novo, 0, arr.length);
		novo[novo.length-1] = elm;
		
		return novo;
	}

}
