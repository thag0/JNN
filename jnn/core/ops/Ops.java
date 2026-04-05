package jnn.core.ops;

import jnn.core.tensor.Tensor;

/**
 * <p>
 * 	Operador interno.
 * </p>
 *		Utilitário auxliar em operações utilizando {@code Tensor}
 * @see jnn.core.tensor.Tensor Tensor
 */
public class Ops {

	/**
	 * Concentração de operações da biblioteca.
	 */
	public Ops() {}

	/**
	 * Realiza a operação {@code  A * B}
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor matmul(Tensor a, Tensor b) {
		return Gemm.matmul(a, b);
	}

	/**
	 * Realiza a operação {@code  A * B}
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @param dst {@code Tensor} de destino.
	 */
	public void matmul(Tensor a, Tensor b, Tensor dst) {
		Gemm.matmul(a, b, dst);
	}

	/**
	 * Realiza a operação de correlação cruzada entre o tensor de entrada e o kernel.
	 * @param x {@code Tensor} de entrada.
	 * @param k {@code Tensor} utilizado para filtro.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor corr2D(Tensor x, Tensor k) {
		return OpsConv.corr2D(x, k);
	}

	/**
	 * Realiza a operação de correlação cruzada entre o tensor de entrada e o kernel.
	 * @param x {@code Tensor} de entrada.
	 * @param k {@code Tensor} utilizado para filtro.
	 * @param dst {@code Tensor} resultado.
	 */
	public void corr2D(Tensor x, Tensor k, Tensor dst) {
		OpsConv.corr2D(x, k, dst);
	}

	/**
	 * Realiza a operação de convolução entre o tensor de entrada e o kernel.
	 * @param x {@code Tensor} de entrada.
	 * @param k {@code Tensor} utilizado para filtro.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor conv2D(Tensor x, Tensor k) {
		return OpsConv.conv2D(x, k);
	}

	/**
	 * Realiza a operação de convolução entre o tensor de entrada e o kernel.
	 * @param x {@code Tensor} de entrada.
	 * @param k {@code Tensor} utilizado para filtro.
	 * @param dst {@code Tensor} resultado.
	 */
	public void conv2D(Tensor x, Tensor k, Tensor dst) {
		OpsConv.conv2D(x, k, dst);
	}

	/**
	 * Realiza a operação de convolução no modo "full" entre o tensor de entrada e o kernel.
	 * @param x {@code Tensor} de entrada.
	 * @param k {@code Tensor} utilizado para filtro.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor conv2DFull(Tensor x, Tensor k) {
		return OpsConv.conv2DFull(x, k);
	}

	/**
	 * Realiza a operação de convolução entre o tensor de entrada e o kernel.
	 * @param entrada {@code Tensor} contendo os dados de entrada.
	 * @param kernel {@code Tensor} contendo o filtro que será aplicado à entrada.
	 * @param saida {@code Tensor} de destino.
	 */
	public void conv2DFull(Tensor entrada, Tensor kernel, Tensor saida) {
		OpsConv.conv2DFull(entrada, kernel, saida);
	}

	/**
	 * Realiza a operação de agrupamento máximo.
	 * @param x {@code Tensor} de entrada.
	 * @param filtro formato do filtro (altura, largura)
	 * @return {@code Tensor} resultado.
	 */
	public Tensor maxPool2D(Tensor x, int[] filtro) {
		return OpsPooling.maxPool2D(x, filtro);
	}

	/**
	 * Realiza a operação de agrupamento máximo.
	 * @param x {@code Tensor} de entrada.
	 * @param filtro formato do filtro (altura, largura)
	 * @param stride formato dos strides (altura, largura)
	 * @return {@code Tensor} resultado.
	 */
	public Tensor maxPool2D(Tensor x, int[] filtro, int[] stride) {
		return OpsPooling.maxPool2D(x, filtro, stride);
	}

	/**
	 * Realiza a operação de agrupamento máximo.
	 * @param x {@code Tensor} de entrada.
	 * @param dst {@code Tensor} destino do resultado.
	 * @param filtro formato do filtro (altura, largura)
	 * @param stride formato dos strides (altura, largura)
	 */
	public void maxPool2D(Tensor x, Tensor dst, int[] filtro, int[] stride) {
		OpsPooling.maxPool2D(x, dst, filtro, stride);
	}

	/**
	 * Realiza a operação de agrupamento médio.
	 * @param x {@code Tensor} de entrada.
	 * @param filtro formato do filtro (altura, largura)
	 * @return {@code Tensor} resultado.
	 */
	public Tensor avgPool2D(Tensor x, int[] filtro) {
		return OpsPooling.avgPool2D(x, filtro);
	}

	/**
	 * Realiza a operação de agrupamento médio.
	 * @param x {@code Tensor} de entrada.
	 * @param filtro formato do filtro (altura, largura)
	 * @param stride formato dos strides (altura, largura)
	 * @return {@code Tensor} resultado.
	 */
	public Tensor avgPool2D(Tensor x, int[] filtro, int[] stride) {
		return OpsPooling.avgPool2D(x, filtro, stride);
	}

	/**
	 * Realiza a operação de agrupamento médio.
	 * @param x {@code Tensor} de entrada.
	 * @param dst {@code Tensor} de destino do resultado.
	 * @param filtro formato do filtro (altura, largura)
	 * @param stride formato dos strides (altura, largura)
	 */
	public void avgPool2D(Tensor x, Tensor dst, int[] filtro, int[] stride) {
		OpsPooling.avgPool2D(x, dst, filtro, stride);
	}
	
}
