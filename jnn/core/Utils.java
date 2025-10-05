package jnn.core;

import java.lang.reflect.Array;
import java.util.Random;

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
	 *arr.shapeStr -> "(2, 3)"
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
	public Tensor[] arrayParaTensores(double[] array) {
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
	public Tensor[] arrayParaTensores(double[][] array) {
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
	public Tensor[] arrayParaTensores(double[][][] array) {
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
	public Tensor[] arrayParaTensores(double[][][][] array) {
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

	/**
	 * Separa um sub conjunto dos dados do array de acordo com os índices fornecidos.
	 * @param <T> tipo de dados do array.
	 * @param arr conjunto de dados desejado.
	 * @param inicio indice inicial (inclusivo).
	 * @param fim índice final (exclusivo).
	 * @return sub conjunto dos dados fornecidos.
	 */
    public <T> T[] subArray(T[] arr, int inicio, int fim) {
        if (inicio < 0 || fim > arr.length || inicio >= fim) {
            throw new IllegalArgumentException(
				"\nÍndices de início ou fim inválidos, (i = " + inicio + ", f = " + fim + ")."
			);
        }

        int tamanho = fim - inicio;

        @SuppressWarnings("unchecked")
        T[] subArr = (T[]) Array.newInstance(arr.getClass().getComponentType(), tamanho);
        System.arraycopy(arr, inicio, subArr, 0, tamanho);

        return subArr;
    }

	/**
	 * Unifica o array em um único {@code Tensor}.
	 * @param ts array de {@code Tensor}.
	 * @return {@code Tensor} concatenado.
	 */
	public Tensor concatenar(Tensor[] ts) {
		int batch = ts.length;

		int[] shape = ts[0].shape();
		int[] novoShape = new int[shape.length+1];
		novoShape[0] = batch;
		for (int i = 0; i < shape.length; i++) {
			novoShape[i+1] = shape[i];
		}

		Tensor c = new Tensor(novoShape);
		for (int i = 0; i < batch; i++) {
			c.subTensor(i).copiar(ts[i]);
		}

		return c;
	}

	/**
	 * Embaralha o array usando o algoritmo Fisher-Yates.
	 * @param <T> tipo de dados de entrada e saida.
	 * @param arr {@code array} base.
	 * @param rng gerador de números aleatórios base.
	 */
	public <T> void embaralhar(T[] arr, Random rng) {
		int n = arr.length;
		Random r = rng == null ? new Random() : rng;
		
		T temp;
		int i, idRng;
		for (i = n - 1; i > 0; i--) {
			idRng = r.nextInt(i+1);
			temp = arr[i];
			arr[i] = arr[idRng];
			arr[idRng] = temp;
		}
	}

	/**
	 * Embaralha o array usando o algoritmo Fisher-Yates.
	 * @param <T> tipo de dados de entrada e saida.
	 * @param arr1 {@code array} 1.
	 * @param arr2 {@code array} 2.
	 * @param rng gerador de números aleatórios base.
	 */
	public <T> void embaralhar(T[] arr1, T[] arr2, Random rng) {
		int n = arr1.length;
		Random r = rng == null ? new Random() : rng;
		
		T temp;
		int i, idRng;
		for (i = n - 1; i > 0; i--) {
			idRng = r.nextInt(i+1);
			
			temp = arr1[i];
			arr1[i] = arr1[idRng];
			arr1[idRng] = temp;

			temp = arr2[i];
			arr2[i] = arr2[idRng];
			arr2[idRng] = temp;
		}
	}

	/**
	 * Converte o objeto recebido em um tensor.
	 * @param obj objeto desejado.
	 * @return {@code Tensor} com base nos dados do objeto.
	 */
	public Tensor paraTensor(Object obj) {
		if (obj instanceof Tensor) {
			return (Tensor) obj;
		}
		
		Tensor t = null;

		if (obj instanceof double[]) {
			double[] arr = (double[]) obj;
			t = new Tensor(arr, arr.length);
		
		} else if (obj instanceof double[][]) {
			t = new Tensor((double[][]) obj);
		
		} else if (obj instanceof double[][][]) {
			t = new Tensor((double[][][]) obj);

		} else if (obj instanceof double[][][][]) {
			t = new Tensor((double[][][][]) obj);

		} else if (obj instanceof double[][][][][]) {
			t = new Tensor((double[][][][][]) obj);
		
		} else {
			throw new IllegalArgumentException(
				"\nTipo de dado \"" + obj.getClass().getTypeName() + "\"" +
				" não suportado."
			);
		}

		return t;
	}

}
