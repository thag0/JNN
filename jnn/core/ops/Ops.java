package jnn.core.ops;

import jnn.core.tensor.Tensor;

/**
 * <p>
 * 	Operador interno.
 * </p>
 *		Utilitário auxliar em operações utilizando {@code Tensor}
 * @see jnn.core.tensor.Tensor Tensor
 */
public abstract class Ops {

	/**
	 * Construtor privado.
	 */
	protected Ops() {}

	/**
	 * Retorna as implementações em {@code CPU} da biblioteca.
	 * @return {@code Backend} em CPU.
	 */
	public static Ops get() {
		return new OpsCPU();
	}

	/**
	 * Realiza a operação {@code  A * B}
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @return {@code Tensor} resultado.
	 */
	public abstract Tensor matmul(Tensor a, Tensor b);

	/**
	 * Realiza a operação {@code  A * B}
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @param dst {@code Tensor} de destino.
	 */
	public abstract void matmul(Tensor a, Tensor b, Tensor dst);

	/**
	 * Realiza a operação de correlação cruzada entre o tensor de entrada e o kernel.
	 * @param x {@code Tensor} de entrada.
	 * @param k {@code Tensor} utilizado para filtro.
	 * @return {@code Tensor} resultado.
	 */
	public abstract Tensor corr2D(Tensor x, Tensor k);

	/**
	 * Realiza a operação de correlação cruzada entre o tensor de entrada e o kernel.
	 * @param x {@code Tensor} de entrada.
	 * @param k {@code Tensor} utilizado para filtro.
	 * @param dst {@code Tensor} resultado.
	 */
	public abstract void corr2D(Tensor x, Tensor k, Tensor dst);

	/**
	 * Método especial para camadas convolucionais.
	 * @param dataX conjunto de dados de entrada.
	 * @param offX offset dos dados de entrada.
	 * @param dataK conjunto de dados do kernel.
	 * @param offK offset dos dados do kernel.
	 * @param dataDst conjunto de dados de destino.
	 * @param offDst offset dos dados de destino.
	 * @param W largura da entrada.
	 * @param H altura da entrada.
	 * @param kW largura do kernel.
	 * @param kH altura do kernel.
	 */
	public abstract void corr2D(
		float[] dataX, int offX, 
		float[] dataK, int offK, 
		float[] dataDst, int offDst, 
		int W, int H, 
		int kW, int kH
	);

	/**
	 * Realiza a operação de convolução entre o tensor de entrada e o kernel.
	 * @param x {@code Tensor} de entrada.
	 * @param k {@code Tensor} utilizado para filtro.
	 * @return {@code Tensor} resultado.
	 */
	public abstract Tensor conv2D(Tensor x, Tensor k);

	/**
	 * Realiza a operação de convolução entre o tensor de entrada e o kernel.
	 * @param x {@code Tensor} de entrada.
	 * @param k {@code Tensor} utilizado para filtro.
	 * @param dst {@code Tensor} resultado.
	 */
	public abstract void conv2D(Tensor x, Tensor k, Tensor dst);

	/**
	 * Realiza a operação de convolução no modo "full" entre o tensor de entrada e o kernel.
	 * @param x {@code Tensor} de entrada.
	 * @param k {@code Tensor} utilizado para filtro.
	 * @return {@code Tensor} resultado.
	 */
	public abstract Tensor conv2DFull(Tensor x, Tensor k);

	/**
	 * Realiza a operação de convolução entre o tensor de entrada e o kernel.
	 * @param entrada {@code Tensor} contendo os dados de entrada.
	 * @param kernel {@code Tensor} contendo o filtro que será aplicado à entrada.
	 * @param saida {@code Tensor} de destino.
	 */
	public abstract void conv2DFull(Tensor entrada, Tensor kernel, Tensor saida);

	/**
	 * Realiza a operação de convolução da entrada {@code X} utilizando o 
	 * kernel {@code K}, no modo "full".
	 * @param dataX {@code array} contendo os dados de entrada.
	 * @param offX offset do array dos dados de entrada.
	 * @param dataK {@code array} contendo os dados do kernel.
	 * @param offK offset do array dos dados do kernel.
	 * @param dataDst {@code array} contendo os dados de destino.
	 * @param offDst offset do array dos dados de destino.
	 * @param W largura da entrada.
	 * @param H altura da entrada.
	 * @param kW largura do kernel.
	 * @param kH altura do kernel.
	 */
	public abstract void conv2DFull(
		float[] dataX, int offX,
		float[] dataK, int offK,
		float[] dataDst, int offDst,
		int W, int H,
		int kW, int kH
	);

	/**
	 * Realiza a operação de agrupamento máximo.
	 * @param x {@code Tensor} de entrada.
	 * @param filtro formato do filtro (altura, largura)
	 * @return {@code Tensor} resultado.
	 */
	public abstract Tensor maxPool2D(Tensor x, int[] filtro);

	/**
	 * Realiza a operação de agrupamento máximo.
	 * @param x {@code Tensor} de entrada.
	 * @param filtro formato do filtro (altura, largura)
	 * @param stride formato dos strides (altura, largura)
	 * @return {@code Tensor} resultado.
	 */
	public abstract Tensor maxPool2D(Tensor x, int[] filtro, int[] stride);

	/**
	 * Realiza a operação de agrupamento máximo.
	 * @param x {@code Tensor} de entrada.
	 * @param dst {@code Tensor} destino do resultado.
	 * @param filtro formato do filtro (altura, largura)
	 * @param stride formato dos strides (altura, largura)
	 */
	public abstract void maxPool2D(Tensor x, Tensor dst, int[] filtro, int[] stride);

	/**
	 * Realiza a operação de agrupamento médio.
	 * @param x {@code Tensor} de entrada.
	 * @param filtro formato do filtro (altura, largura)
	 * @return {@code Tensor} resultado.
	 */
	public abstract Tensor avgPool2D(Tensor x, int[] filtro);

	/**
	 * Realiza a operação de agrupamento médio.
	 * @param x {@code Tensor} de entrada.
	 * @param filtro formato do filtro (altura, largura)
	 * @param stride formato dos strides (altura, largura)
	 * @return {@code Tensor} resultado.
	 */
	public abstract Tensor avgPool2D(Tensor x, int[] filtro, int[] stride);

	/**
	 * Realiza a operação de agrupamento médio.
	 * @param x {@code Tensor} de entrada.
	 * @param dst {@code Tensor} de destino do resultado.
	 * @param filtro formato do filtro (altura, largura)
	 * @param stride formato dos strides (altura, largura)
	 */
	public abstract void avgPool2D(Tensor x, Tensor dst, int[] filtro, int[] stride);
	
}
