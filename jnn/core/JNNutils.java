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
	public static int ultimoIndice(int[] arr) {
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
	public static boolean apenasMaiorZero(int[] arr) {
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
	public static boolean apenasMaiorZero(float[] arr) {
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
	 * @param array array desejado.
	 * @return array de {@code Tensores}.
	 */
	public static Tensor[] arrayParaTensores(float[] array) {
		int n = array.length;

		Tensor[] arr = new Tensor[n];
		for (int i = 0; i < n; i++) {
			arr[i] = new Tensor(1).set(array[i], 0);
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
	public static Tensor[] arrayParaTensores(float[][] array) {
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
	public static Tensor[] arrayParaTensores(float[][][] array) {
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
	public static Tensor[] arrayParaTensores(float[][][][] array) {
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
	public static void validarNaoNulo(Object obj, String msg) {
		if (obj == null) {
			String str;

			if (msg == null) {
				str = "obj == null.";
				
			} else {
				msg = msg.trim();
				str = msg.isEmpty() ? "obj == null." : msg;
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
	public static <T> T[] addEmArray(T[] arr, T elm) {
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
    public static <T> T[] subArray(T[] arr, int inicio, int fim) {
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
	public static Tensor concatenar(Tensor[] ts) {
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
	 * @param r gerador de números aleatórios base.
	 */
	public static <T> void embaralhar(T[] arr, Random r) {
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
	public static <T> void embaralhar(T[] arr1, T[] arr2, Random r) {
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
	public static Tensor paraTensor(Object obj) {
		return TensorConverter.tensor(obj);
	}

    /**
     * Formata um valor de tamanho em bytes para um formato mais fácil de ler.
     * @param bytes quantidade total em bytes.
     * @return valor formatado.
     */
    public static String formatarTamBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        int exp = (int) (Math.log(bytes) / Math.log(1024));
        char prefixo = "KMGTPE".charAt(exp - 1); // K, M, G, T, P, E
        
        return String
        .format("%.2f %sB", bytes / Math.pow(1024, exp), prefixo)
        .replaceAll(",", ".");
    }

}
