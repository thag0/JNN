package jnn.core;

import java.util.Optional;

import jnn.core.tensor.Tensor;

/**
 * <h2>
 * 	Operador de Tensores
 * </h2>
 * Utilitário auxliar em operações utilizando {@code Tensor}
 * <p>
 * Implementação completa: {@link https://github.com/thag0/JNN}
 * </p>
 * @see {@link jnn.core.tensor.Tensor}
 */
public class OpTensor {
	
	/**
	 * Auxiliar em operações utilizando {@code Tensor}.
	 * @see {@link Tensor}
	 */
	public OpTensor() {}

	/**
	 * Realiza a operação {@code A + B}.
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor matadd(Tensor a, Tensor b) {
		if (!a.compShape(b)) {
			throw new IllegalArgumentException(
				"\nDimensões do tensor A " + a.shapeStr() + 
				" e B " + b.shapeStr() + " devem ser iguais."
			);
		}

		if (a.numDim() != 2) {
			throw new UnsupportedOperationException(
				"\nOs tensores devem ter duas dimensões"
			);
		}

		return a.map(b, (x, y) -> x + y);
	}

	/**
	 * Realiza a operação {@code A - B}.
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor matsub(Tensor a, Tensor b) {
		if (!a.compShape(b)) {
			throw new IllegalArgumentException(
				"\nDimensões do tensor A " + a.shapeStr() + 
				" e B " + b.shapeStr() + " devem ser iguais."
			);
		}

		if (a.numDim() != 2) {
			throw new UnsupportedOperationException(
				"\nOs tensores devem ter duas dimensões"
			);
		}

		return a.map(b, (x, y) -> x - y);
	}

	/**
	 * Realiza a operação {@code A ⊙ B}.
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor mathad(Tensor a, Tensor b) {
		if (!a.compShape(b)) {
			throw new IllegalArgumentException(
				"\nDimensões do tensor A " + a.shapeStr() + 
				" e B " + b.shapeStr() + " devem ser iguais."
			);
		}

		if (a.numDim() != 2) {
			throw new UnsupportedOperationException(
				"\nOs tensores devem ter duas dimensões"
			);
		}

		return a.map(b, (x, y) -> x * y);
	}

	/**
	 * Realiza a operação {@code A / B}.
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor matdiv(Tensor a, Tensor b) {
		if (!a.compShape(b)) {
			throw new IllegalArgumentException(
				"\nDimensões do tensor A " + a.shapeStr() + 
				" e B " + b.shapeStr() + " devem ser iguais."
			);
		}

		if (a.numDim() != 2) {
			throw new UnsupportedOperationException(
				"\nOs tensores devem ter duas dimensões"
			);
		}

		return a.map(b, (x, y) -> x / y);
	}

	/**
	 * Realiza a operação {@code  A * B}
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor matmul(Tensor a, Tensor b) {
		if (a.numDim() > 2 || b.numDim() > 2) {
			throw new IllegalArgumentException(
				"\nOs tensores devem conter até duas dimensões, mas contêm " +
				"A = " + a.numDim() + " B = " + b.numDim()
			);
		}
	
		int[] shapeA = a.shape();
		int[] shapeB = b.shape();

		int linA = shapeA.length == 1 ? 1 : shapeA[0];
		int colA = shapeA.length == 1 ? shapeA[0] : shapeA[1];
		int linB = shapeB.length == 1 ? 1 : shapeB[0];
		int colB = shapeB.length == 1 ? shapeB[0] : shapeB[1];
	
		if (colA != linB) {
			throw new IllegalArgumentException(
				"As dimensões dos tensores não são compatíveis para multiplicação de matrizes: " +
				"A = " + a.shapeStr() + " B = " + b.shapeStr()
			);
		}
	
		Tensor res = linA == 1 ? new Tensor(colB) : new Tensor(linA, colB);
		
		matmul(a, b, res);
	
		return res;
	}

	/**
	 * Realiza a operação {@code  A * B}
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @param dst {@code Tensor} de destino.
	 */
	public void matmul(Tensor a, Tensor b, Tensor dst) {
		if (a.numDim() > 2 || b.numDim() > 2 || dst.numDim() > 2) {
			throw new IllegalArgumentException(
				"\nOs tensores devem conter até duas dimensões, mas contêm " +
				"A = " + a.numDim() + " B = " + b.numDim() + " Dest = " + dst.numDim()
			);
		}

		final int[] shapeA = a.shape();
		final int[] shapeB = b.shape();
		final int[] shapeD = dst.shape();

		final int linA = shapeA.length == 1 ? 1 : shapeA[0];
		final int colA = shapeA.length == 1 ? shapeA[0] : shapeA[1];
		final int linB = shapeB.length == 1 ? 1 : shapeB[0];
		final int colB = shapeB.length == 1 ? shapeB[0] : shapeB[1];
		final int linD = shapeD.length == 1 ? 1 : shapeD[0];
		final int colD = shapeD.length == 1 ? shapeD[0] : shapeD[1];
	
		if (colA != linB) {
			throw new IllegalArgumentException(
				"As dimensões dos tensores não são compatíveis para multiplicação de matrizes: " +
				"A = " + a.shapeStr() + " B = " + b.shapeStr()
			);
		}

		if (linA != linD || colB != colD) {
			throw new IllegalArgumentException(
				"\nDimensões de saída inesperadas, esperado (" + linA + ", " + colB +  ")" +
				", mas recebido " + dst.shapeStr() 
			);
		}
	
		final int[] stridesA = a.strides();
		final int[] stridesB = b.strides();
		final int[] stridesD = dst.strides();

		final int s0A = stridesA.length == 1 ? 1 : stridesA[0];
		final int s1A = stridesA.length == 1 ? 1 : stridesA[1];
		final int s0B = stridesB.length == 1 ? 1 : stridesB[0];
		final int s1B = stridesB.length == 1 ? 1 : stridesB[1];
		final int s0D = stridesD.length == 1 ? 1 : stridesD[0];
		final int s1D = stridesD.length == 1 ? 1 : stridesD[1];

		final int offsetA = a.offset();
		final int offsetB = b.offset();
		final int offsetD = dst.offset();

		final double[] dataA = a.array();
		final double[] dataB = b.array();
		final double[] dataD = dst.array();

		for (int i = 0; i < linA; i++) {
			final int baseA = offsetA + i * s0A;
			final int baseD = offsetD + i * s0D;

			for (int k = 0; k < colA; k++) {
				final double valA = dataA[baseA + k * s1A];
				final int baseB = offsetB + k * s0B;

				for (int j = 0; j < colB; j++) {
					dataD[baseD + j * s1D] += valA * dataB[baseB + j * s1B];
				}
			}
		}

	}

	/**
	 * Experimental
	 * @param x {@code Tensor} de entrada.
	 * @param altK altura do kernel (filtro).
	 * @param largK largura do kernel (filtro).
	 * @param altStd altura do stride.
	 * @param largStd largura do stride.
	 * @param altPad altura do padding.
	 * @param largPad largura do padding.
	 * @return {@code Tensor} convertido para o formato {@code im2col}.
	 */
	public Tensor im2col(Tensor x, int altK, int largK, int altStd, int largStd, int altPad, int largPad) {
		int[] shape = x.shape();
		if (shape.length != 3) {
			throw new IllegalArgumentException("O tensor de entrada deve ter formato [C, H, W].");
		}

		int canais = shape[0];
		int altIn = shape[1];
		int largIn = shape[2];

		int altOut = (altIn + 2 * altPad - altK) / altStd + 1;
		int largOut = (largIn + 2 * largPad - largK) / largStd + 1;

		Tensor col = new Tensor(new int[]{canais * altK * largK, altOut * largOut});
		double[] dadosX = x.array();
		double[] dadosC = col.array();

		int canalArea = altK * largK;
		int colunaCol = altOut * largOut;

		// Iteração principal
		for (int c = 0; c < canais; c++) {
			for (int kh = 0; kh < altK; kh++) {
				for (int kw = 0; kw < largK; kw++) {
					int linhaBase = c * canalArea + kh * largK + kw;
					int outIndex = 0;
					int inY0 = kh - altPad;

					for (int y = 0; y < altOut; y++) {
						int inY = inY0 + y * altStd;
						int inX0 = kw - largPad;

						for (int x2 = 0; x2 < largOut; x2++) {
							int inX = inX0 + x2 * largStd;

							if (inY >= 0 && inY < altIn && inX >= 0 && inX < largIn) {
								dadosC[linhaBase * colunaCol + outIndex] = dadosX[c * altIn * largIn + inY * largIn + inX];
							} else {
								dadosC[linhaBase * colunaCol + outIndex] = 0.0;
							}

							outIndex++;
						}
					}
				}
			}
		}

		return col;
	}

	/**
	 * Calcula o formato de saída de operações convolucionais.
	 * @param entrada formato de entrada {@code (altura, largura)}.
	 * @param filtro formato do filtro aplicado {@code (altura, largura)}.
	 * @param stride formato dos strides {@code (altura, largura)}.
	 * @return formato de saída calculado {@code (altura, largura)}..
	 */
	private int[] calcShapeConv(int[] entrada, int[] filtro, int[] stride) {
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
	 * Realiza a operação de correlação cruzada entre o tensor de entrada e o kernel.
	 * @param x {@code Tensor} de entrada.
	 * @param k {@code Tensor} utilizado para filtro.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor corr2D(Tensor x, Tensor k) {
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
	 * Realiza a operação de correlação cruzada entre o tensor de entrada e o kernel.
	 * @param x {@code Tensor} de entrada.
	 * @param k {@code Tensor} utilizado para filtro.
	 * @return {@code Tensor} resultado.
	 */
	public void corr2D(Tensor x, Tensor k, Tensor dst) {
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

		double[] dataE = x.array();
		double[] dataK = k.array();
		double[] dataS = dst.array();

		// lidar com views de tensores
		int offE = x.offset();
		int offK = k.offset();
		int offS = dst.offset();

		double soma;
		for (int i = 0; i < altEsp; i++) {
			for (int j = 0; j < largEsp; j++) {

				soma = 0.0;
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
	 * Realiza a operação de convolução entre o tensor de entrada e o kernel.
	 * @param x {@code Tensor} de entrada.
	 * @param k {@code Tensor} utilizado para filtro.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor conv2D(Tensor x, Tensor k) {
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
	 * Realiza a operação de convolução entre o tensor de entrada e o kernel.
	 * @param x {@code Tensor} de entrada.
	 * @param k {@code Tensor} utilizado para filtro.
	 * @return {@code Tensor} resultado.
	 */
	public void conv2D(Tensor x, Tensor k, Tensor dst) {
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

		double[] dataE = x.array();
		double[] dataK = k.array();
		double[] dataS = dst.array();

		int offE = x.offset();
		int offK = k.offset();
		int offS = dst.offset();
	
		double soma;
		for (int i = 0; i < altEsp; i++) {
			for (int j = 0; j < largEsp; j++) {
				
				soma = 0.0;
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
	 * Realiza a operação de convolução no modo "full" entre o tensor de entrada e o kernel.
	 * @param x {@code Tensor} de entrada.
	 * @param k {@code Tensor} utilizado para filtro.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor conv2DFull(Tensor x, Tensor k) {
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
	 * Realiza a operação de convolução entre o tensor de entrada e o kernel.
	 * @param entrada {@code Tensor} contendo os dados de entrada.
	 * @param kernel {@code Tensor} contendo o filtro que será aplicado à entrada.
	 * @param saida {@code Tensor} de destino.
	 */
	public void conv2DFull(Tensor entrada, Tensor kernel, Tensor saida) {
		int[] shapeE = entrada.shape();
		int[] shapeK = kernel.shape();
		int[] shapeS = saida.shape();

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
				" recebido " + saida.shapeStr()
			);
		}

		int altEntrada = shapeE[0];
		int largEntrada = shapeE[1];
		int altKernel = shapeK[0];
		int largKernel = shapeK[1];

		double[] dataE = entrada.array();
		double[] dataK = kernel.array();
		double[] dataS = saida.array();

		// lidar com views de tensores
		int offE = entrada.offset();
		int offK = kernel.offset();
		int offS = saida.offset();

		double soma;
		for (int i = 0; i < altEsp; i++) {
			for (int j = 0; j < largEsp; j++) {
				
				soma = 0.0;
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
	 * Realiza a operação de agrupamento máximo.
	 * @param x {@code Tensor} de entrada.
	 * @param filtro formato do filtro (altura, largura)
	 * @return {@code Tensor} resultado.
	 */
	public Tensor maxPool2D(Tensor x, int[] filtro) {
		return maxPool2D(x, filtro, filtro);// stride = filtro
	}

	/**
	 * Realiza a operação de agrupamento máximo.
	 * @param x {@code Tensor} de entrada.
	 * @param filtro formato do filtro (altura, largura)
	 * @param stride formato dos strides (altura, largura)
	 * @return {@code Tensor} resultado.
	 */
	public Tensor maxPool2D(Tensor x, int[] filtro, int[] stride) {
		if (x.numDim() != 2) {
			throw new IllegalArgumentException(
				"\nEntrada deve ser 2D, mas é " + x.numDim() + "D."
			);
		}

		if (filtro.length != 2) {
			throw new IllegalArgumentException(
				"\nFormato do filtro deve conter dois elementos, " +
				" recebido " + filtro.length
			);
		}

		if (stride.length != 2) {
			throw new IllegalArgumentException(
				"\nFormato do stride deve conter dois elementos, " +
				" recebido " + filtro.length
			);
		}

		int[] shapeEntrada = x.shape();

		int[] poolShape = calcShapeConv(
			shapeEntrada, 
			filtro, 
			stride
		);

		Tensor pool = new Tensor(poolShape[0], poolShape[1]);
		maxPool2D(x, pool, filtro, stride);

		return pool;
	}

	/**
	 * Realiza a operação de agrupamento máximo.
	 * @param x {@code Tensor} de entrada.
	 * @param dst {@code Tensor} destino do resultado.
	 * @param filtro formato do filtro (altura, largura)
	 * @param stride formato dos strides (altura, largura)
	 */
	public void maxPool2D(Tensor x, Tensor dst, int[] filtro, int[] stride) {
		if (x.numDim() != 2 || dst.numDim() != 2) {
			throw new IllegalArgumentException(
				"maxPool2D agora aceita apenas tensores 2D. Entrada=" +
				x.numDim() + "D, Saída=" + dst.numDim() + "D."
			);
		}

		if (filtro.length != 2) {
			throw new IllegalArgumentException("Filtro deve ser [fH, fW]");
		}

		if (stride.length != 2) {
			throw new IllegalArgumentException("Stride deve ser [sH, sW]");
		}

		int H = x.shape()[0];
		int W = x.shape()[1];

		int H_out = dst.shape()[0];
		int W_out = dst.shape()[1];

		int[] esperado = calcShapeConv(new int[]{H, W}, filtro, stride);
		if (esperado[0] != H_out || esperado[1] != W_out) {
			throw new IllegalArgumentException(
				"Saída esperada = [" + esperado[0] + "," + esperado[1] +
				"] mas recebeu " + dst.shapeStr()
			);
		}

		double[] arrIn  = x.array();
		double[] arrOut = dst.array();

		int baseOffsetIn  = x.offset();
		int baseOffsetOut = dst.offset();

		int inStrideH  = x.strides()[0];
		int inStrideW  = x.strides()[1];
		int outStrideH = dst.strides()[0];
		int outStrideW = dst.strides()[1];

		int fH = filtro[0];
		int fW = filtro[1];
		int sH = stride[0];
		int sW = stride[1];

		double maxVal, val;

		for (int i = 0; i < H_out; i++) {
			int linInicio = i * sH;
			int linFim = Math.min(linInicio + fH, H);

			for (int j = 0; j < W_out; j++) {
				int colInicio = j * sW;
				int colFim = Math.min(colInicio + fW, W);
				maxVal = Double.NEGATIVE_INFINITY;

				for (int l = linInicio; l < linFim; l++) {
					int baseLinha = baseOffsetIn + l * inStrideH;
					for (int m = colInicio; m < colFim; m++) {
						val = arrIn[baseLinha + m * inStrideW];
						if (val > maxVal) maxVal = val;
					}
				}

				arrOut[baseOffsetOut + i * outStrideH + j * outStrideW] = maxVal;
			}
		}
	}

	/**
	 * Realiza a operação de agrupamento médio.
	 * @param x {@code Tensor} de entrada.
	 * @param filtro formato do filtro (altura, largura)
	 * @return {@code Tensor} resultado.
	 */
	public Tensor avgPool2D(Tensor x, int[] stride) {
		return avgPool2D(x, stride, stride);// stride = filtro
	}

	/**
	 * Realiza a operação de agrupamento médio.
	 * @param x {@code Tensor} de entrada.
	 * @param filtro formato do filtro (altura, largura)
	 * @param stride formato dos strides (altura, largura)
	 * @return {@code Tensor} resultado.
	 */
	public Tensor avgPool2D(Tensor x, int[] filtro, int[] stride) {
		if (x.numDim() != 2) {
			throw new IllegalArgumentException(
				"\nEntrada deve ser 2D, mas é " + x.numDim() + "D."
			);
		}

		if (filtro.length != 2) {
			throw new IllegalArgumentException(
				"\nFormato do filtro deve conter dois elementos, " +
				" recebido " + filtro.length
			);
		}

		if (stride.length != 2) {
			throw new IllegalArgumentException(
				"\nFormato do stride deve conter dois elementos, " +
				" recebido " + filtro.length
			);
		}

		int[] shapeEntrada = x.shape();

		int[] poolShape = calcShapeConv(
			shapeEntrada, 
			filtro, 
			stride
		);

		Tensor pool = new Tensor(poolShape[0], poolShape[1]);
		avgPool2D(x, pool, filtro, stride);

		return pool;		
	}

	/**
	 * Realiza a operação de agrupamento médio.
	 * @param x {@code Tensor} de entrada.
	 * @param dst {@code Tensor} de destino do resultado.
	 * @param filtro formato do filtro (altura, largura)
	 * @param stride formato dos strides (altura, largura)
	 */
	public void avgPool2D(Tensor x, Tensor dst, int[] filtro, int[] stride) {
		if (x.numDim() != 2 || dst.numDim() != 2) {
			throw new UnsupportedOperationException(
				"\nAmbos os tensores devem ser 2D, recebido " +
				" entrada = " + x.numDim() + "D e saida = " + dst.numDim() + "D."
			);
		}

		if (filtro.length != 2) {
			throw new IllegalArgumentException(
				"\nFormato do filtro deve conter dois elementos, " +
				" recebido " + filtro.length
			);
		}

		if (stride.length != 2) {
			throw new IllegalArgumentException(
				"\nFormato do stride deve conter dois elementos, " +
				" recebido " + filtro.length
			);
		}

		int[] shapeEntrada = x.shape();
		int[] shapeSaida   = dst.shape();

		int altEntrada  = shapeEntrada[0];
		int largEntrada = shapeEntrada[1];
		int altSaida    = shapeSaida[0];
		int largSaida   = shapeSaida[1];

		int[] shapeEsp = calcShapeConv(
			shapeEntrada, 
			filtro,
			stride
		);
		
		if (altSaida != shapeEsp[0] || largSaida != shapeEsp[1]) {
			throw new IllegalArgumentException(
				"\nDimensão de saída esperada (" + shapeEsp[0] + ", " + shapeEsp[1] + "), mas" +
				" recebido " + dst.shapeStr()
			);
		}

		double[] dataE = x.array();
		double[] dataS = dst.array();

		int offE = x.offset();
		int[] strE = x.strides();

		int offS = dst.offset();
		int[] strS = dst.strides();

		int fH = filtro[0];
		int fW = filtro[1];
		int sH = stride[0];
		int sW = stride[1];

		for (int i = 0; i < altSaida; i++) {
			int linInicio = i * sH;
			int linFim = Math.min(linInicio + fH, altEntrada);

			for (int j = 0; j < largSaida; j++) {
				int colInicio = j * sW;
				int colFim = Math.min(colInicio + fW, largEntrada);

				double soma = 0;
				int cont = 0;
				for (int l = linInicio; l < linFim; l++) {
					int baseLinha = offE + l * strE[0];
					for (int m = colInicio; m < colFim; m++) {
						soma += dataE[baseLinha + m * strE[1]];
						cont++;
					}
				}
				
				dataS[offS + i * strS[0] + j * strS[1]] = soma / cont;
			}
		}
		
	}
	
	/**
	 * Experimental
	 * @param entrada
	 * @param kernel
	 * @param bias
	 * @param saida
	 */
	public void forwardConv2DIm2col(Tensor entrada, Tensor kernel, Optional<Tensor> bias, Tensor saida) {
		int[] kShape = kernel.shape();// (filtros, canais, kH, kW)
		int numFiltros = kShape[0];
		int canais = kShape[1];
		int kH = kShape[2];
		int kW = kShape[3];
		int padH = 0; 
		int padW = 0;
		int strideH = 1;
		int strideW = 1;

		int H = entrada.tamDim(1);
		int W = entrada.tamDim(2);
		int outH = (H + 2 * padH - kH) / strideH + 1;
		int outW = (W + 2 * padW - kW) / strideW + 1;

		Tensor im2Col = im2col(entrada, kH, kW, strideH, strideW, padH, padW);
		Tensor flatK = kernel.reshape(numFiltros, canais * kH * kW); 

		Tensor res = new Tensor(numFiltros, outH * outW);
		matmul(flatK, im2Col, res);

		res = res.reshape(numFiltros, outH, outW);
		saida.copiar(res);
		
		bias.ifPresent(b -> {
			for (int f = 0; f < numFiltros; f++) {
				double x = b.get(f);
				saida.subTensor(f).add(x);
			}
		});
	}
	
}
