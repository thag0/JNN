package jnn.core.ops;

import jnn.core.tensor.Tensor;

/**
 * Implementações internas de operações de convolução.
 */
public class OpsConv {

	/**
	 * Construtor privado.
	 */
	private OpsConv() {}
    
	private static int[] calcShapeConv(int[] entrada, int[] filtro, int[] stride) {
		if (entrada.length != 2 || filtro.length != 2 || stride.length != 2) {
			throw new IllegalArgumentException(
				"\nTodos os formatos devem conter dois elementos (altura, largura)."
			);
		}

		return new int[] {
			(int) Math.floor((float)(entrada[0] - filtro[0]) / stride[0]) + 1,
			(int) Math.floor((float)(entrada[1] - filtro[1]) / stride[1]) + 1
		};
	}

	/**
	 * Realiza a operação de correlaão cruzada da entrada {@code X} 
	 * utilizando o kernel {@code K}.
	 * @param x {@code Tensor} de entrada.
	 * @param k {@code Tensor} contendo kernel.
	 * @return {@code Tensor} resultado.
	 */
	public static Tensor corr2D(Tensor x, Tensor k) {
		if (x.numDim() != 2 || k.numDim() != 2) {
			throw new IllegalArgumentException(
				"\nAmbos os tensores devem ter duas dimensões."
			);

		}

		int[] shapeE = x.shape();
		int[] shapeK = k.shape();
		
		Tensor corr = new Tensor(calcShapeConv(shapeE, shapeK, new int[] {1, 1}));

		corr2D(x, k, corr);

		return corr;
	}

	/**
	 * Realiza a operação de correlaão cruzada da entrada {@code X} 
	 * utilizando o kernel {@code K}.
	 * @param x {@code Tensor} de entrada.
	 * @param k {@code Tensor} contendo kernel.
	 * @param dst {@code Tensor} resultado.
	 */
	public static void corr2D(Tensor x, Tensor k, Tensor dst) {
		int[] shapeE = x.shape();
		int[] shapeK = k.shape();
		int[] shapeS = dst.shape();

		if (shapeE.length != 2 || shapeK.length != 2 || shapeS.length != 2) {
			throw new IllegalArgumentException(
				"\nTodos os tensores devem ter duas dimensões, mas Entrada " + shapeE.length + "D, " +
				" Kernel " + shapeK.length + "D e Saida " + shapeS.length + "D." 
			);
		}

		int altEsp  = shapeE[0] - shapeK[0] + 1;
		int largEsp = shapeE[1] - shapeK[1] + 1;
	
		int altSaida = shapeS[0];
		int largSaida = shapeS[1];
		if (altSaida != altEsp || largSaida != largEsp) {
			throw new IllegalArgumentException(
				"\nDimensão de saída esperada (" + altEsp + ", " + largEsp + "), mas" +
				" recebido " + dst.shapeStr()
			);
		}

		int altKernel = shapeK[0];
		int largKernel = shapeK[1];
		int largEntrada = shapeE[1];

		float[] dataE = x.array();
		float[] dataK = k.array();
		float[] dataS = dst.array();

		int offE = x.offset();
		int offK = k.offset();
		int offS = dst.offset();

		float soma;
		for (int i = 0; i < altEsp; i++) {
			for (int j = 0; j < largEsp; j++) {

				soma = 0.0f;
				final int idSaida = offS + (i * largEsp + j);
				for (int l = 0; l < altKernel; l++) {
					int idBaseEntrada = offE + ((l + i) * largEntrada);
					int idBaseKernel  = offK + (l * largKernel); 
					for (int m = 0; m < largKernel; m++) {
						soma += 
							dataE[idBaseEntrada + (m + j)] *
							dataK[idBaseKernel + m];
					}
				}

				dataS[idSaida] += soma;
			}
		}

	}

	/**
	 * Realiza a operação de correlaão cruzada da entrada {@code X} 
	 * utilizando o kernel {@code K}.
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
	public static void corr2D(
		float[] dataX, 
		int offX,
		float[] dataK, 
		int offK,
		float[] dataDst, 
		int offDst,
		int W, 
		int H,
		int kW, 
		int kH
	) {
		final int outH = H - kH + 1;
		final int outW = W - kW + 1;

		for (int i = 0; i < outH; i++) {
			int baseOut = offDst + i * outW;
			int baseIn  = offX   + i * W;

			for (int j = 0; j < outW; j++) {
				float soma = 0.0f;

				int inColBase = baseIn + j;

				for (int kh = 0; kh < kH; kh++) {
					int inRow = inColBase + kh * W;
					int kRow  = offK + kh * kW;

					for (int kw = 0; kw < kW; kw++) {
						soma += dataX[inRow + kw] * dataK[kRow + kw];
					}
				}

				dataDst[baseOut + j] += soma;
			}
		}
	}

	/**
	 * Realiza a operação de convolução da entrada {@code X} utilizando o 
	 * kernel {@code K}.
	 * @param x {@code Tensor} de entrada.
	 * @param k {@code Tensor} contendo kernel.
	 * @return {@code Tensor} resultado.
	 */
	public static Tensor conv2D(Tensor x, Tensor k) {
		if (x.numDim() != 2 || k.numDim() != 2) {
			throw new IllegalArgumentException(
				"\nTodos os tensores devem ter duas dimensões."
			);
		}

		int[] shapeE = x.shape();
		int[] shapeK = k.shape();
		
		int[] shapeS = calcShapeConv(shapeE, shapeK, new int[] {1, 1});
		Tensor conv = new Tensor(shapeS);

		conv2D(x, k, conv);

		return conv;
	}

	/**
	 * Realiza a operação de convolução da entrada {@code X} utilizando o 
	 * kernel {@code K}.
	 * @param x {@code Tensor} de entrada.
	 * @param k {@code Tensor} contendo kernel.
	 * @param dst {@code Tensor} resultado.
	 */
	public static void conv2D(Tensor x, Tensor k, Tensor dst) {
		int[] shapeE = x.shape();
		int[] shapeK = k.shape();
		int[] shapeS = dst.shape();

		if (shapeE.length != 2 || shapeK.length != 2 || shapeS.length != 2) {
			throw new IllegalArgumentException(
				"\nTodos os tensores devem ter duas dimensões, mas Entrada " + shapeE.length + "D, " +
				" Kernel " + shapeK.length + "D e Saida " + shapeS.length + "D." 
			);
		}
		
		int altEsp  = shapeE[0] - shapeK[0] + 1;
		int largEsp = shapeE[1] - shapeK[1] + 1;
	
		int altSaida = shapeS[0];
		int largSaida = shapeS[1];
		if (altSaida != altEsp || largSaida != largEsp) {
			throw new IllegalArgumentException(
				"\nDimensão de saída esperada (" + altEsp + ", " + largEsp + "), mas" +
				" recebido " + dst.shapeStr()
			);
		}
	
		int altKernel = shapeK[0];
		int largKernel = shapeK[1];
		int largEntrada = shapeE[1];

		float[] dataE = x.array();
		float[] dataK = k.array();
		float[] dataS = dst.array();

		int offE = x.offset();
		int offK = k.offset();
		int offS = dst.offset();
	
		float soma;
		for (int i = 0; i < altEsp; i++) {
			for (int j = 0; j < largEsp; j++) {
				
				soma = 0.0f;
				final int idSaida = offS + (i * largEsp + j);
				for (int l = 0; l < altKernel; l++) {
					for (int m = 0; m < largKernel; m++) {
						soma +=
							dataE[offE + ((l + i) * largEntrada + (m + j))] * 
							dataK[offK + ((altKernel - 1 - l) * largKernel + (largKernel - 1 - m))];
					}
				}

				dataS[idSaida] += soma;
			}
		}
		
	}

	/**
	 * Realiza a operação de convolução da entrada {@code X} utilizando o 
	 * kernel {@code K}, no modo "full".
	 * @param x {@code Tensor} de entrada.
	 * @param k {@code Tensor} contendo kernel.
	 * @return {@code Tensor} resultado.
	 */
	public static Tensor conv2DFull(Tensor x, Tensor k) {
		if (x.numDim() != 2 || k.numDim() != 2) {
			throw new IllegalArgumentException(
				"\nAmbos os tensores devem ter duas dimensões."
			);

		}

		int[] shapeE = x.shape();
		int[] shapeK = k.shape();
		
		int alt  = shapeE[0] + shapeK[0] - 1;
		int larg = shapeE[1] + shapeK[1] - 1;
	
		Tensor conv = new Tensor(alt, larg);

		conv2DFull(x, k, conv);
	
		return conv;
	}

	/**
	 * Realiza a operação de convolução da entrada {@code X} utilizando o 
	 * kernel {@code K}, no modo "full".
	 * @param x {@code Tensor} de entrada.
	 * @param k {@code Tensor} contendo kernel.
	 * @param dst {@code Tensor} resultado.
	 */
	public static void conv2DFull(Tensor x, Tensor k, Tensor dst) {
		int[] shapeE = x.shape();
		int[] shapeK = k.shape();
		int[] shapeS = dst.shape();

		if (shapeE.length > 2 || shapeK.length > 2 || shapeS.length > 2) {
			throw new IllegalArgumentException(
				"\nTodos os tensores devem ter duas dimensões, mas Entrada " + shapeE.length + "D, " +
				" Kernel " + shapeK.length + "D e Saida " + shapeS.length + "D." 
			);
		}
		
		int altEsp  = shapeE[0] + shapeK[0] - 1;
		int largEsp = shapeE[1] + shapeK[1] - 1;
	
		int altSaida  = shapeS[0];
		int largSaida = shapeS[1];
		if (altSaida != altEsp || largSaida != largEsp) {
			throw new IllegalArgumentException(
				"\nDimensão de saída esperada (" + altEsp + ", " + largEsp + "), mas" +
				" recebido " + dst.shapeStr()
			);
		}

		int altEntrada = shapeE[0];
		int largEntrada = shapeE[1];
		int altKernel = shapeK[0];
		int largKernel = shapeK[1];

		float[] dataE = x.array();
		float[] dataK = k.array();
		float[] dataS = dst.array();

		// lidar com views de tensores
		int offE = x.offset();
		int offK = k.offset();
		int offS = dst.offset();

		float soma;
		for (int i = 0; i < altEsp; i++) {
			for (int j = 0; j < largEsp; j++) {
				
				soma = 0.0f;
				final int idSaida = offS + (i * largEsp + j);
				for (int m = 0; m < altKernel; m++) {
					int linEntrada = i - m;
					if (linEntrada >= 0 && linEntrada < altEntrada) {
						for (int n = 0; n < largKernel; n++) {
							int colEntrada = j - n;
							if (colEntrada >= 0 && colEntrada < largEntrada) {
								soma +=
									dataK[offK + (m * largKernel + n)] *
									dataE[offE + (linEntrada * largEntrada + colEntrada)];
							}
						}
					}
				}

				dataS[idSaida] += soma;
			}
		}

	}

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
	public static void conv2DFull(float[] dataX, int offX, float[] dataK, int offK, float[] dataDst, int offDst, int W, int H, int kW, int kH) {
		final int outH = H + kH - 1;
		final int outW = W + kW - 1;

		for (int i = 0; i < outH; i++) {
			final int baseOut = offDst + i * outW;

			for (int j = 0; j < outW; j++) {
				float soma = 0.0f;

				for (int kh = 0; kh < kH; kh++) {
					int inRow = i - kh;
					if (inRow < 0 || inRow >= H) continue;

					final int baseIn = offX + inRow * W;
					final int baseK  = offK + kh * kW;

					for (int kw = 0; kw < kW; kw++) {
						int inCol = j - kw;
						if (inCol < 0 || inCol >= W) continue;

						soma += dataK[baseK + kw] *
							dataX[baseIn + inCol];
					}
				}

				dataDst[baseOut + j] += soma;
			}
		}
	}


}
