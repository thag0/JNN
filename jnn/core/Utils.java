package jnn.core;

import jnn.core.tensor.Tensor4D;

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
	 * Desserializa o array no array de matrizes de destino.
	 * @param arr array contendo os dados.
	 * @param dest destino da cópia
	 */
	public void copiar(double[] arr, Mat[] dest) {
		if (arr.length != (dest.length * dest[0].tamanho())) {
			throw new IllegalArgumentException(
				"Tamanhos incompatíveis entre o array (" + arr.length + 
				") e o destino (" + (dest.length * dest[0].tamanho()) + ")."
			);
		}
	
		int id = 0, i, j, k;
		for (i = 0; i < dest.length; i++) {
			for (j = 0; j < dest[i].lin(); j++) {
				for (k = 0; k < dest[i].col(); k++) {
					dest[i].editar(j, k, arr[id++]);
				}
			}
		}
	}

	/**
	 * Desserializa o array na matriz de matriz de destino.
	 * @param arr array contendo os dados.
	 * @param dest destino da cópia
	 */
	public void copiar(double[] arr, Mat[][] dest) {
		if (arr.length != (dest.length * dest[0].length * dest[0][0].tamanho())) {
			throw new IllegalArgumentException(
				"Tamanhos incompatíveis entre o array (" + arr.length + 
				") e o destino (" + (dest.length * dest[0].length * dest[0][0].tamanho()) + ")."
			);
		}

		int id = 0;
		int i, j, k, l;
		for (i = 0; i < dest.length; i++) {
			for (j = 0; j < dest[i].length; j++) {
				for (k = 0; k < dest[i][j].lin(); k++) {
					for (l = 0; l < dest[i][j].col(); l++) {
						dest[i][j].editar(k, l, arr[id++]);
					}
				}
			}
		}
	}

	/**
	 * Copia o conteúdo contido do array de matrizes para o
	 * destino desejado.
	 * @param arr array de matrizes contendo os dados.
	 * @param dest destino da cópia.
	 */
	public void copiar(double[][][] arr, Mat[] dest) {
		if ((arr.length * arr[0].length * arr[0][0].length) != (dest.length * dest[0].tamanho())) {
			throw new IllegalArgumentException(
				"Tamanhos incompatíveis entre o tensor (" + (arr.length * arr[0].length) + 
				") e o destino (" + (dest.length * dest[0].tamanho()) + ")."
			);
		}

		for (int i = 0; i < dest.length; i++) {
			dest[i].copiar(arr[i]);
		}
	}

	/**
	 * Copia o conteúdo contido do array de matrizes para o
	 * destino desejado.
	 * @param arr array de matrizes contendo os dados.
	 * @param dest destino da cópia.
	 */
	public void copiar(Mat[] arr, Mat[] dest) {
		for (int i = 0; i < dest.length; i++) {
			dest[i].copiar(arr[i]);
		}
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
	 * Transforma a entrada num array de amostras.
	 * <p>
	 *    Até o momento criei esse método para usar com os tensores.
	 * </p>
	 * @param entrada dados de entrada, se já for um array ele será preservado.
	 * Caso seja um tensor, será convertido num array de sub tensores.
	 * @return array de objetos para uso no treino.
	 */
	public Object[] transformarParaArray(Object entrada) {
		Object[] elementos = new Object[0];

		if (entrada instanceof Object[]) {
			Object[] arr = (Object[]) entrada;
			//clone para corrigir inconsistências
			elementos = arr.clone();
		
		} else if (entrada instanceof Tensor4D) {
			Tensor4D t = (Tensor4D) entrada;
			int idArray = 0;
			int[] dim = t.shape();

			for (int i = dim.length-1; i >= 0; i--) {
				if (dim[i] > 1) idArray = i;
			}

			Tensor4D[] amostras = new Tensor4D[dim[idArray]];

			for (int i = 0; i < amostras.length; i++) {
				switch (idArray) {
					case 0://tensores 3d
						amostras[i] = new Tensor4D(t.array3D(i)); 
						break;

					case 1://matrizes
						amostras[i] = new Tensor4D(t.array2D(0, i));
						break;

					case 2://vetores
						amostras[i] = new Tensor4D(1, 1, 1, dim[idArray+1]);
						for (int j = 0; j < amostras[i].dim3(); j++) {
							amostras[i].copiar(t.array1D(0, 0, i), 0, 0, j);
						}
						break;

					case 3://escalar
						amostras[i] = new Tensor4D(1, 1, 1, 1);
						amostras[i].set(t.get(0, 0, 0, i), 0, 0, 0, 0);
						break;
				
					default:
						break;
				}

				amostras[i].nome("amostra " + i);//ajudar no debug
			}

			elementos = new Object[amostras.length];
			System.arraycopy(amostras, 0, elementos, 0, amostras.length);
		
		} else {
			throw new IllegalArgumentException(
				"Tipo de objeto (" + entrada.getClass().getSimpleName() + ") inválido."
			);
		}

		return elementos;
	}
}
