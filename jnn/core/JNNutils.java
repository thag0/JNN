package jnn.core;

import java.lang.reflect.Array;
import java.util.Random;

import jnn.core.tensor.Tensor;
import jnn.core.tensor.TensorConverter;

/**
 * Utilitário geral para a biblioteca.
 */
public final class JNNutils {

	/**
	 * Gerador de números aleatórios.
	 */
	private static Random rng = new Random();

	/**
	 * Utilitário geral para a biblioteca.
	 */
	private JNNutils() {}

	/**
	 * Configura a seed do gerador de números aleatórios.
	 * @param seed nova seed.
	 */
	public static void randSeed(long seed) {
		rng.setSeed(seed);
	}

	/**
	 * Retorna um valor aleatório no intervalor [min, max[.
	 * @param min valor mínimo (inclusivo).
	 * @param max valor máximo (exclusivo).
	 * @return valor gerado.
	 */
	public static int randInt(int min, int max) {
		return rng.nextInt(min, max);
	}

	/**
	 * Retorna um valor aleatório no intervalo de [0, max[.
	 * @param max valor máximo (exclusivo).
	 * @return valor gerado.
	 */
	public static int randInt(int max) {
		return rng.nextInt(0, max);
	}

	/**
	 * Retorna um número normalmente distribuído com média = 0 
	 * e desvio padrão = 1.
	 * @return valor gerado.
	 */
	public static float randGaussianf() {
		return (float) rng.nextGaussian();
	}

	/**
	 * Retorna um número aleatório uniformemente distribuído com 
	 * média = 0 e desvio padrão = 1.
	 * @return valor gerado.
	 */
	public static float randFloat() {
		return rng.nextFloat();
	}

	/**
	 * Retorna um número aleatório de acordo com o intervalo especificado.
	 * @param min valor mínimo (inclusivo).
	 * @param max valor valor máximo (exclusivo).
	 * @return valor gerado.
	 */
	public static float randFloat(float min, float max) {
		return rng.nextFloat(min, max);
	}

	/**
	 * Retorna um número normalmente distribuído com média = 0 
	 * e desvio padrão = 1.
	 * @return valor gerado.
	 */
	public static double randGaussian() {
		return rng.nextGaussian();
	}

	/**
	 * Retorna um número aleatório uniformemente distribuído com 
	 * média = 0 e desvio padrão = 1.
	 * @return valor gerado.
	 */
	public static double randDouble() {
		return rng.nextDouble();
	}

	/**
	 * Retorna um número aleatório de acordo com o intervalo especificado.
	 * @param min valor mínimo (inclusivo).
	 * @param max valor valor máximo (exclusivo).
	 * @return valor gerado.
	 */
	public static double randDouble(double min, double max) {
		return rng.nextDouble(min, max);
	}

	/**
	 * Retorna o último índice válido do array.
	 * @param arr array base.
	 * @return último índice.
	 */
	public static int lastIndex(int[] arr) {
		if (arr.length == 0) {
			throw new IllegalArgumentException("\nArray de tamanho 0.");
		}

		return arr.length-1;
	}

	/**
	 * Verifica se o conteúdo do array contém valores maiores que zero.
	 * @param arr array base.
	 * @return resultado da verificação.
	 */
	public static boolean isAllPositive(int[] arr) {
		for (int i = 0; i < arr.length; i++) {
			if (arr[i] < 1) return false;
		}

		return true;
	}

	/**
	 * Verifica se o conteúdo do array contém valores maiores que zero.
	 * @param arr array base.
	 * @return resultado da verificação.
	 */
	public static boolean isAllPositive(float[] arr) {
		for (int i = 0; i < arr.length; i++) {
			if (arr[i] < 0) return false;
		}

		return true;
	}

	/**
	 * Compara se os dois arrays são igual em tamanho e numericamente.
	 * @param a {@code array} 1.
	 * @param b {@code array} 2.
	 * @return {@code true} se os arrays são igual, {@code false} caso contrário.
	 */
	public static boolean arrayComp(int[] a, int[] b) {
		if (a.length != b.length) return false;

		int n = a.length;
		for (int i = 0; i < n; i++) {
			if (a[i] != b[i]) return false;
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
	public static String arrayStr(int[] arr) {
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
	 * @param arr array desejado.
	 * @return array de {@code Tensores}.
	 */
	public static Tensor[] arrayToTensors(float[] arr) {
		int n = arr.length;

		Tensor[] newArr = new Tensor[n];
		for (int i = 0; i < n; i++) {
			newArr[i] = new Tensor(1).set(arr[i], 0);
			newArr[i].name("amostra " + i);// ajudar no debug
		}

		return newArr;
	}

	/**
	 * Converte um array primitivo para um array de tensores, cada tensor contendo
	 * um {@code array1D}.
	 * @param arr array desejado.
	 * @return array de {@code Tensores}.
	 */
	public static Tensor[] arrayToTensors(float[][] arr) {
		int lin = arr.length;
		int col = arr[0].length;

		Tensor[] newArr = new Tensor[lin];
		for (int i = 0; i < lin; i++) {
			newArr[i] = new Tensor(col);
			newArr[i].copyElements(arr[i]);
			newArr[i].name("amostra " + i);// ajudar no debug
		}

		return newArr;
	}

	/**
	 * Converte um array primitivo para um array de tensores, cada tensor contendo
	 * um {@code array2D}.
	 * @param arr array desejado.
	 * @return array de {@code Tensores}.
	 */
	public static Tensor[] arrayToTensors(float[][][] arr) {
		Tensor[] newArr = new Tensor[arr.length];

		int n = arr.length;
		for (int i = 0; i < n; i++) {
			newArr[i] = new Tensor(arr[i]);
			newArr[i].name("amostra " + i);// ajudar no debug
		}

		return newArr;	
	}

	/**
	 * Converte um array primitivo para um array de tensores, cada tensor contendo
	 * um {@code array3D}.
	 * @param arr array desejado.
	 * @return array de {@code Tensores}.
	 */
	public static Tensor[] arrayToTensors(float[][][][] arr) {
		Tensor[] newArr = new Tensor[arr.length];

		int n = arr.length;
		for (int i = 0; i < n; i++) {
			newArr[i] = new Tensor(arr[i]);
			newArr[i].name("amostra " + i);// ajudar no debug
		}

		return newArr;
	}

	/**
	 * Verifica se o objeto recebido é nulo.
	 * @param <T> tipo do objeto.
	 * @param obj objeto de comparação.
	 * @param msg mensagem personalizada de erro.
	 * @throws NullPointerException caso o objeto seja nulo.
	 * @return objeto caso não seja nulo.
	 */
	public static <T> T validateNonNull(T obj, String msg) {
		if (obj == null) {
			throw new NullPointerException(
				"\n" + (msg == null || msg.isBlank() ? "obj == null." : msg.trim())
			);
		}

		return obj;
	}

	/**
	 * Adiciona um novo elemento ao array.
	 * @param <T> tipo dos elementos do array
	 * @param arr {@code array}.
	 * @param e elemento para adição.
	 * @return novo array com elemento adicionado.
	 */
	public static <T> T[] addToArray(T[] arr, T e) {
		validateNonNull(arr, "arr = null");
		validateNonNull(e, "e = null");
		
		@SuppressWarnings("unchecked")
		T[] novo = (T[]) Array.newInstance(arr.getClass().getComponentType(), arr.length + 1);

		System.arraycopy(arr, 0, novo, 0, arr.length);
		novo[novo.length-1] = e;
		
		return novo;
	}

	/**
	 * Separa um sub conjunto dos dados do array de acordo com os índices fornecidos.
	 * @param <T> tipo de dados do array.
	 * @param arr conjunto de dados desejado.
	 * @param start indice inicial (inclusivo).
	 * @param end índice final (exclusivo).
	 * @return sub conjunto dos dados fornecidos.
	 */
    public static <T> T[] subArray(T[] arr, int start, int end) {
        if (start < 0 || end > arr.length || start >= end) {
            throw new IllegalArgumentException(
				"\nÍndices de início ou fim inválidos, (i = " + start + ", f = " + end + ")."
			);
        }

        int tamanho = end - start;

        @SuppressWarnings("unchecked")
        T[] subArr = (T[]) Array.newInstance(arr.getClass().getComponentType(), tamanho);
        System.arraycopy(arr, start, subArr, 0, tamanho);

        return subArr;
    }

	/**
	 * Unifica o array em um único {@code Tensor}.
	 * @param ts array de {@code Tensor}.
	 * @return {@code Tensor} concatenado.
	 */
	public static Tensor concat(Tensor[] ts) {
		int batch = ts.length;

		int[] shape = ts[0].shape();
		int[] novoShape = new int[shape.length+1];
		novoShape[0] = batch;
		for (int i = 0; i < shape.length; i++) {
			novoShape[i+1] = shape[i];
		}

		Tensor c = new Tensor(novoShape);
		for (int i = 0; i < batch; i++) {
			c.subTensor(i).copy(ts[i]);
		}

		return c;
	}

	/**
	 * Embaralha o array usando o algoritmo Fisher-Yates.
	 * @param <T> tipo de dados de entrada e saida.
	 * @param arr {@code array} base.
	 * @param r gerador de números aleatórios base.
	 */
	public static <T> void shuffle(T[] arr, Random r) {
		int n = arr.length;
		Random rand = r == null ? rng : r;
		
		T temp;
		int i, idRng;
		for (i = n - 1; i > 0; i--) {
			idRng = rand.nextInt(i+1);
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
	 * @param r gerador de números aleatórios base.
	 */
	public static <T> void shuffle(T[] arr1, T[] arr2, Random r) {
		int n = arr1.length;
		Random rand = r == null ? rng : r;
		
		T temp;
		int i, idRng;
		for (i = n - 1; i > 0; i--) {
			idRng = rand.nextInt(i+1);
			
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
	public static Tensor toTensor(Object obj) {
		return TensorConverter.tensor(obj);
	}

    /**
     * Formata um valor de tamanho em bytes para um formato mais fácil de ler.
     * @param bytes quantidade total em bytes.
     * @return valor formatado.
     */
    public static String formatSizeBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        int exp = (int) (Math.log(bytes) / Math.log(1024));
        char prefixo = "KMGTPE".charAt(exp - 1); // K, M, G, T, P, E
        
        return String
        .format("%.2f %sB", bytes / Math.pow(1024, exp), prefixo)
        .replaceAll(",", ".");
    }

}
