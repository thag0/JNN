package jnn.core;

import java.util.Optional;

import jnn.core.tensor.Tensor;
import jnn.core.tensor.Variavel;

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
				"A = " + a.shapeStr() + " B = " + b.shapeStr()
			);
		}
	
		int[] shapeA = a.shape();
		int[] shapeB = b.shape();

		final int linA = shapeA.length == 1 ? 1 : shapeA[0];
		final int colA = shapeA.length == 1 ? shapeA[0] : shapeA[1];
		final int linB = shapeB.length == 1 ? 1 : shapeB[0];
		final int colB = shapeB.length == 1 ? shapeB[0] : shapeB[1];
	
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
		int[] shapeA = a.shape();
		int[] shapeB = b.shape();
		int[] shapeD = dst.shape();

		if (shapeA.length > 2 || shapeB.length > 2 || shapeD.length > 2) {
			throw new IllegalArgumentException(
				"\nOs tensores devem conter até duas dimensões, mas contêm " +
				"A = " + a.shapeStr() + " B = " + b.shapeStr() + " Dest = " + dst.shapeStr()
			);
		}

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
	
		int[] stridesA = a.strides();
		int[] stridesB = b.strides();
		int[] stridesD = dst.strides();

		// Se for vetor, ajusta strides para 1D
		int s0A = shapeA.length == 1 ? 1 : stridesA[0];
		int s1A = shapeA.length == 1 ? 1 : stridesA[1];
		int s0B = shapeB.length == 1 ? 1 : stridesB[0];
		int s1B = shapeB.length == 1 ? 1 : stridesB[1];
		int s0D = shapeD.length == 1 ? 1 : stridesD[0];
		int s1D = shapeD.length == 1 ? 1 : stridesD[1];

		Variavel[] dataA = a.paraArray();
		Variavel[] dataB = b.paraArray();
		Variavel[] dataD = dst.paraArray();
		double soma;

		for (int i = 0; i < linD; i++) {
			for (int j = 0; j < colD; j++) {
				soma = 0;
				for (int k = 0; k < colA; k++) {
					soma += 
						dataA[(i * s0A) + (k * s1A)].get() * 
						dataB[(k * s0B) + (j * s1B)].get();
				}
				dataD[(i * s0D) + (j * s1D)].add(soma);
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
		int[] xShape = x.shape();
		int C = xShape[0];
		int H = xShape[1];
		int W = xShape[2];

		int outH = (H + 2 * altPad - altK) / altStd + 1;
		int outW = (W + 2 * largPad - largK) / largStd + 1;

		// col = (C * kH * kW, outH * outW)
		Tensor col = new Tensor(C * altK * largK, outH * outW);

		Variavel[] dataX = x.paraArray();
		Variavel[] dataC = col.paraArray();
		int[] stdX = x.strides();
		int[] stdC = col.strides();

		for (int c = 0; c < C; c++) {
			int cAjuste = c * stdX[0];

			for (int ak = 0; ak < altK; ak++) {//Altura Kernel
				for (int lk = 0; lk < largK; lk++) {//Largura Kernel
					int lin = c * altK * largK + ak * largK + lk;

					for (int as = 0; as < outH; as++) {//Altura Saída
						int ih = as * altStd - altPad + ak;

						for (int ls = 0; ls < outW; ls++) {//Largura Saída
							int iw = ls * largStd - largPad + lk;
							int colId = as * outW + ls;

							int colAjuste = lin * stdC[0] + colId * stdC[1];

							if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
								int xAjuste = cAjuste + ih * stdX[1] + iw * stdX[2];
								dataC[colAjuste].set(dataX[xAjuste].get());
							} else {
								dataC[colAjuste].set(0.0);
							}
						}
					}
				}
			}
		}

		return col.nome("im2col");
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

		Variavel[] dataE = x.paraArray();
		Variavel[] dataK = k.paraArray();
		Variavel[] dataS = dst.paraArray();

		Variavel soma = new Variavel();// mais rápido que usar double
		for (int i = 0; i < altEsp; i++) {
			for (int j = 0; j < largEsp; j++) {

				soma.zero();
				final int idSaida = i * largEsp + j;
				for (int l = 0; l < altKernel; l++) {
					int idBaseEntrada = (l + i) * largEntrada;
					int idBaseKernel  = l * largKernel; 
					for (int m = 0; m < largKernel; m++) {
						soma.addmul(
							dataE[idBaseEntrada + (m + j)],
							dataK[idBaseKernel + m]
						);
					}
				}

				dataS[idSaida].add(soma);
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

		Variavel[] dataE = x.paraArray();
		Variavel[] dataK = k.paraArray();
		Variavel[] dataS = dst.paraArray();
	
		Variavel soma = new Variavel();
		for (int i = 0; i < altEsp; i++) {
			for (int j = 0; j < largEsp; j++) {
				
				soma.zero();
				final int idSaida = i * largEsp + j;
				for (int l = 0; l < altKernel; l++) {
					for (int m = 0; m < largKernel; m++) {
						soma.addmul(
							dataE[(l + i) * largEntrada + (m + j)], 
							dataK[(altKernel - 1 - l) * largKernel + (largKernel - 1 - m)]
						);
					}
				}

				dataS[idSaida].add(soma);
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

		Variavel[] dataE = entrada.paraArray();
		Variavel[] dataK = kernel.paraArray();
		Variavel[] dataS = saida.paraArray();

		Variavel soma = new Variavel();
		for (int i = 0; i < altEsp; i++) {
			for (int j = 0; j < largEsp; j++) {
				
				soma.zero();
				final int idSaida = i*largEsp + j;
				for (int m = 0; m < altKernel; m++) {
					int linEntrada = i - m;
					if (linEntrada >= 0 && linEntrada < altEntrada) {
						for (int n = 0; n < largKernel; n++) {
							int colEntrada = j - n;
							if (colEntrada >= 0 && colEntrada < largEntrada) {
								soma.addmul(
									dataK[m * largKernel + n],
									dataE[linEntrada * largEntrada + colEntrada]
								);
							}
						}
					}
				}

				dataS[idSaida].add(soma);
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
		if (x.numDim() != 3) {
			throw new IllegalArgumentException(
				"\nEntrada deve ser 3D, mas é " + x.numDim() + "D."
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
			new int[] {shapeEntrada[1], shapeEntrada[2]}, 
			filtro, 
			stride
		);

		Tensor pool = new Tensor(shapeEntrada[0], poolShape[0], poolShape[1]);
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
		if (x.numDim() != 3 || dst.numDim() != 3) {
			throw new UnsupportedOperationException(
				"\nAmbos os tensores devem ser 3D, recebido " +
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
		int[] shapeSaida = dst.shape();

		int canais = shapeEntrada[0];
		int altEntrada  = shapeEntrada[1];
		int largEntrada = shapeEntrada[2];
		int altSaida  = shapeSaida[1];
		int largSaida = shapeSaida[2];

		int[] shapeEsp = calcShapeConv(
			new int[]{ altEntrada, largEntrada}, 
			filtro,
			stride
		);
		
		if (altSaida != shapeEsp[0] || largSaida != shapeEsp[1]) {
			throw new IllegalArgumentException(
				"\nDimensão de saída esperada (" + shapeEsp[0] + ", " + shapeEsp[1] + "), mas" +
				" recebido " + dst.shapeStr()
			);
		}

		Variavel[] dataE = x.paraArray();
		Variavel[] dataS = dst.paraArray();

		int canalSizeEntrada = altEntrada * largEntrada;
		int canalSizeSaida   = altSaida   * largSaida;
		double maxVal, val;

		for (int c = 0; c < canais; c++) {
			int baseEntrada = c * canalSizeEntrada;
			int baseSaida   = c * canalSizeSaida;

			for (int i = 0; i < altSaida; i++) {
				int linInicio = i * stride[0];
				int linFim = Math.min(linInicio + filtro[0], altEntrada);

				for (int j = 0; j < largSaida; j++) {
					int colInicio = j * stride[1];
					int colFim = Math.min(colInicio + filtro[1], largEntrada);

					maxVal = Double.NEGATIVE_INFINITY;

					for (int l = linInicio; l < linFim; l++) {
						int idLinha = baseEntrada + l * largEntrada;
						for (int m = colInicio; m < colFim; m++) {
							val = dataE[idLinha + m].get();
							if (val > maxVal) maxVal = val;
						}
					}

					dataS[baseSaida + i * largSaida + j].set(maxVal);
				}
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
		if (x.numDim() != 3) {
			throw new IllegalArgumentException(
				"\nEntrada deve ser 3D, mas é " + x.numDim() + "D."
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
			new int[] {shapeEntrada[1], shapeEntrada[2]}, 
			filtro, 
			stride
		);

		Tensor pool = new Tensor(shapeEntrada[0], poolShape[0], poolShape[1]);
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
		if (x.numDim() != 3 || dst.numDim() != 3) {
			throw new UnsupportedOperationException(
				"\nAmbos os tensores devem ser 3D, recebido " +
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

		int canais      = shapeEntrada[0];
		int altEntrada  = shapeEntrada[1];
		int largEntrada = shapeEntrada[2];
		int altSaida    = shapeSaida[1];
		int largSaida   = shapeSaida[2];

		int[] shapeEsp = calcShapeConv(
			new int[]{ altEntrada, largEntrada}, 
			filtro,
			stride
		);
		
		if (altSaida != shapeEsp[0] || largSaida != shapeEsp[1]) {
			throw new IllegalArgumentException(
				"\nDimensão de saída esperada (" + shapeEsp[0] + ", " + shapeEsp[1] + "), mas" +
				" recebido " + dst.shapeStr()
			);
		}

		Variavel[] dataE = x.paraArray();
		Variavel[] dataS = dst.paraArray();

		int canalSizeEntrada = altEntrada * largEntrada;
		int canalSizeSaida   = altSaida   * largSaida;

		for (int c = 0; c < canais; c++) {
			int baseEntrada = c * canalSizeEntrada;
			int baseSaida   = c * canalSizeSaida;

			for (int i = 0; i < altSaida; i++) {
				int linInicio = i * stride[0];
				int linFim    = Math.min(linInicio + filtro[0], altEntrada);

				for (int j = 0; j < largSaida; j++) {
					int colInicio = j * stride[1];
					int colFim    = Math.min(colInicio + filtro[1], largEntrada);

					double soma = 0;
					int cont = 0;

					for (int l = linInicio; l < linFim; l++) {
						int idLinha = baseEntrada + l * largEntrada;
						for (int m = colInicio; m < colFim; m++) {
							soma += dataE[idLinha + m].get();
							cont++;
						}
					}

					dataS[baseSaida + i * largSaida + j].set(soma / cont);
				}
			}
		}
	}

	/**
	 * Realiza a peopagação direta através da camada Densa.
	 * @param entrada {@code Tensor} contendo a entrada da camada.
	 * @param kernel {@code Tensor} contendos o kernel/pesos da camada.
	 * @param bias {@code Tensor} contendo o bias da camada {@code (podendo ser nulo)}.
	 * @param saida {@code Tensor} de destino do resultado.
	 * @see {@link jnn.camadas.Densa}
	 */
	public void forwardDensa(Tensor entrada, Tensor kernel, Optional<Tensor> bias, Tensor saida) {
		matmul(entrada, kernel, saida);

		bias.ifPresent(b -> {
			if (entrada.numDim() == 1) {//amostra única
				saida.add(b);

			} else if (entrada.numDim() == 2) {//lote de amostras
				saida.copiar(
					saida.broadcast(b, Double::sum)
				);
			}
		});
	}

	/**
	 * Realiza a propagação reversa através da camada densa.
	 * @param entrada {@code Tensor} contendo a entrada da camada.
	 * @param kernel {@code Tensor} contendos o kernel/pesos da camada.
	 * @param gradS {@code Tensor} contendo o gradiente em relação a saída da camada.
	 * @param gradK {@code Tensor} contendo o gradiente em relação ao kernel/pesos da camada.
	 * @param gradB {@code Tensor} contendo o gradiente em relação ao bias da camada {@code (podendo ser nulo)}.
	 * @param gradE {@code Tensor} contendo o gradiente em relação à entrada da camada.
	 * @see {@link jnn.camadas.Densa}
	 */
	public void backwardDensa(Tensor entrada, Tensor kernel, Tensor gradS, Tensor gradK, Optional<Tensor> gradB, Tensor gradE) {	
		if (gradS.numDim() == 1) {//amostra única
			matmul(entrada.unsqueeze(0).transpor(), gradS, gradK);
			matmul(gradS, kernel.transpor(), gradE);
			gradB.ifPresent(gb -> gb.add(gradS));
		
		} else if (gradS.numDim() == 2) {//lote de amostras
			matmul(entrada.transpor(), gradS, gradK);
			matmul(gradS, kernel.transpor(), gradE);
			
			int lotes = gradS.tamDim(0);
			gradB.ifPresent(gb -> {
				for (int i = 0; i < lotes; i++) {					
					gb.add(gradS.subTensor(i));
				}
			});
		}

	}

	/**
	 * Realiza a propagação direta através da camada convolucional.
	 * @param entrada {@code Tensor} contendo a entrada da camada.
	 * @param kernel {@code Tensor} contendos o kernel/filtros da camada.
	 * @param bias {@code Tensor} contendo o bias da camada {@code (podendo ser nulo)}.
	 * @param saida {@code Tensor} de destino do resultado.
	 * @see {@link jnn.camadas.Conv2D}
	 */
	public void forwardConv2D(Tensor entrada, Tensor kernel, Optional<Tensor> bias, Tensor saida) {
		int[] shapeK = kernel.shape();
		final int profEntrada = shapeK[1];
		final int numFiltros = shapeK[0];

		// NOTA
		// mesmo paralelizando, não tem ganho.
		// acredito que essa nova abordagem facilite a paralelização
		// em melhorias futuras que sejam mais otimizadas

		Tensor[] entradas = new Tensor[profEntrada];
		for (int i = 0; i < profEntrada; i++) {
			entradas[i] = entrada.subTensor(i);
		}

		Tensor[][] kernels = new Tensor[numFiltros][profEntrada];
		for (int i = 0; i < numFiltros; i++) {
			Tensor ki = kernel.subTensor(i);
			for (int j = 0; j < profEntrada; j++) {
				kernels[i][j] = ki.subTensor(j);
			}
		}

		Tensor[] saidas = new Tensor[numFiltros];
		for (int i = 0; i < numFiltros; i++) {
			saidas[i] = saida.subTensor(i);
		}

		if (bias.isPresent()) {
			Tensor b = bias.get();
			for (int i = 0; i < numFiltros; i++) {
				for (int e = 0; e < profEntrada; e++) {
					corr2D(entradas[e], kernels[i][e], saidas[i]);
				}
				saidas[i].add(b.get(i));
			}

		} else {
			for (int i = 0; i < numFiltros; i++) {
				for (int e = 0; e < profEntrada; e++) {
					corr2D(entradas[e], kernels[i][e], saidas[i]);
				}
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

		int H = entrada.shape()[1];
		int W = entrada.shape()[2];
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

	/**
	 * Realiza a propagação reversa através da camada convolucional.
	 * @param entrada {@code Tensor} contendo a entrada da camada.
	 * @param kernel {@code Tensor} contendos o kernel/filtros da camada.
	 * @param gradS {@code Tensor} contendo o gradiente em relação a saída da camada.
	 * @param gradK {@code Tensor} contendo o gradiente em relação ao kernel/filtros da camada.
	 * @param gradB {@code Tensor} contendo o gradiente em relação ao bias da camada {@code (podendo ser nulo)}.
	 * @param gradE {@code Tensor} contendo o gradiente em relação à entrada da camada.
	 * @see {@link jnn.camadas.Conv2D}
	 */
	public void backwardConv2D(Tensor entrada, Tensor kernel, Tensor gradS, Tensor gradK, Optional<Tensor> gradB, Tensor gradE) {
		int[] shapeK = kernel.shape();

		final int numFiltros = shapeK[0];
		final int profEntrada = shapeK[1];

		Tensor[] gsSaida = new Tensor[numFiltros];
		for (int i = 0; i < numFiltros; i++) {
			gsSaida[i] = gradS.subTensor(i);
		}

		// gradiente em relação as entradas
		Tensor[][] kernels = new Tensor[numFiltros][profEntrada];
		for (int i = 0; i < numFiltros; i++) {
			Tensor ki = kernel.subTensor(i);
			for (int j = 0; j < profEntrada; j++) {
				kernels[i][j] = ki.subTensor(j);
			}
		}

		Tensor[] gsEntrada = new Tensor[profEntrada];
		for (int i = 0; i < profEntrada; i++) {
			gsEntrada[i] = gradE.subTensor(i);
		}

		for (int e = 0; e < profEntrada; e++) {
			for (int f = 0; f < numFiltros; f++) {
				conv2DFull(gsSaida[f], kernels[f][e], gsEntrada[e]);
			}
		}

		// gradiente em relação aos kernels
		Tensor[] entradas = new Tensor[profEntrada];
		for (int i = 0; i < profEntrada; i++) {
			entradas[i] = entrada.subTensor(i);
		}

		Tensor[][] gsKernels = new Tensor[numFiltros][profEntrada];
		for (int i = 0; i < numFiltros; i++) {
			Tensor ki = gradK.subTensor(i);
			for (int j = 0; j < profEntrada; j++) {
				gsKernels[i][j] = ki.subTensor(j);
			}
		}

		for (int f = 0; f < numFiltros; f++) {
			for (int e = 0; e < profEntrada; e++) {
				corr2D(entradas[e], gsSaida[f], gsKernels[f][e]);	
			}
		}

		// gradiente em relação aos bias
		gradB.ifPresent(gb -> {
			for (int i = 0; i < numFiltros; i++) {
				double soma = gradS.subTensor(i).soma().item();
				gb.add(soma, i);
			}
		});
	}

}
