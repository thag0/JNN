package jnn.core.tensor;

/**
 * Auxiliar em operação para tensores.
 */
public class OpTensor {
	
	/**
	 * Auxiliar em operação para tensores 4D.
	 */
	public OpTensor() {}

	/**
	 * Realiza a operação {@code A + B}.
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor matAdd(Tensor a, Tensor b) {
		if (!a.compararShape(b)) {
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
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor matSub(Tensor a, Tensor b) {
		if (!a.compararShape(b)) {
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
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor matHadamard(Tensor a, Tensor b) {
		if (!a.compararShape(b)) {
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
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor matDiv(Tensor a, Tensor b) {
		if (!a.compararShape(b)) {
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
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor matMult(Tensor a, Tensor b) {
		if (a.numDim() > 2 || b.numDim() > 2) {
			throw new IllegalArgumentException(
				"\nOs tensores devem conter até duas dimensões, mas contêm " +
				"A = " + a.shapeStr() + " B = " + b.shapeStr()
			);
		}

		boolean unsqueezeA = false;
		boolean unsqueezeB = false;

		if (a.numDim() == 1) {
			a.unsqueeze(0);
			unsqueezeA = true;
		}

		if (b.numDim() == 1) {
			b.unsqueeze(0);
			unsqueezeB = true;
		}
	
		int[] shapeA = a.shape();
		int[] shapeB = b.shape();

		int linA = shapeA[0];
		int colA = shapeA[1];
		int linB = shapeB[0];
		int colB = shapeB[1];
	
		if (colA != linB) {
			throw new IllegalArgumentException(
				"As dimensões dos tensores não são compatíveis para multiplicação de matrizes: " +
				"A = " + a.shapeStr() + " B = " + b.shapeStr()
			);
		}
	
		Tensor res = new Tensor(linA, colB);
		int n = colA;
		for (int i = 0; i < linA; i++) {
			for (int j = 0; j < colB; j++) {
				double soma = 0.0d;
				for (int k = 0; k < n; k++) {
					soma += a.get(i, k) * b.get(k, j);
				}
				res.set(soma, i, j);
			}
		}

		if (unsqueezeA) {
			a.squeeze(0);
			res.squeeze(0);
		}

		if (unsqueezeB) {
			b.squeeze(0);
		}
	
		return res;
	}

	/**
	 * Realiza a operação {@code  A * B}
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @param dest {@code Tensor} de destino.
	 */
	public void matMult(Tensor a, Tensor b, Tensor dest) {
		if (a.numDim() > 2 || b.numDim() > 2 | dest.numDim() > 2) {
			throw new IllegalArgumentException(
				"\nOs tensores devem conter até duas dimensões, mas contêm " +
				"A = " + a.shapeStr() + " B = " + b.shapeStr() + " Dest = " + dest.shapeStr()
			);
		}

		boolean unsqueezeA = false;
		boolean unsqueezeB = false;
		boolean unsqueezeD = false;

		if (a.numDim() == 1) {
			a.unsqueeze(0);
			unsqueezeA = true;
		}

		if (b.numDim() == 1) {
			b.unsqueeze(0);
			unsqueezeB = true;
		}

		if (dest.numDim() == 1) {
			dest.unsqueeze(0);
			unsqueezeD = true;
		}
	
		int[] shapeA = a.shape();
		int[] shapeB = b.shape();
		int[] shapeD = dest.shape();

		final int linA = shapeA[0];
		final int colA = shapeA[1];
		final int linB = shapeB[0];
		final int colB = shapeB[1];
		final int linD = shapeD[0];
		final int colD = shapeD[1];
	
		if (colA != linB) {
			throw new IllegalArgumentException(
				"As dimensões dos tensores não são compatíveis para multiplicação de matrizes: " +
				"A = " + a.shapeStr() + " B = " + b.shapeStr()
			);
		}

		if (linA != linD || colB != colD) {
			throw new IllegalArgumentException(
				"\nDimensões de saída inesperadas, esperado (" + linA + ", " + colB +  ")" +
				", mas recebido " + dest.shapeStr() 
			);
		}

		//vetorização para melhor performance				
		double[] dataA = a.paraArrayDouble();
		double[] dataB = b.paraArrayDouble();
		Variavel[] dataD = dest.paraArray();
		
		final int n = colA;
		for (int i = 0; i < linA; i++) {
			for (int j = 0; j < colB; j++) {
				double soma = 0.0d;
				for (int k = 0; k < n; k++) {
					soma += dataA[i * colA + k] * dataB[k * colB + j];
				}
				dataD[i * colD + j].set(soma);
			}
		}

		// voltar os tensores para seus formatos originais
		if (unsqueezeA) a.squeeze(0);
		if (unsqueezeB) b.squeeze(0);
		if (unsqueezeD) dest.squeeze(0);
	}

	/**
	 * Realiza a operação de correlação cruzada entre o tensor de entrada e o kernel.
	 * @param entrada {@code Tensor} contendo os dados de entrada.
	 * @param kernel {@code Tensor} contendo o filtro que será aplicado à entrada.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor correlacao2D(Tensor entrada, Tensor kernel) {
		if (entrada.numDim() != 2 || kernel.numDim() != 2) {
			throw new IllegalArgumentException(
				"\nAmbos os tensores devem ter duas dimensões."
			);

		}

		int[] shapeE = entrada.shape();
		int[] shapeK = kernel.shape();
		
		int alt  = shapeE[0] - shapeK[0] + 1;
		int larg = shapeE[1] - shapeK[1] + 1;
		
		Tensor saida = new Tensor(alt, larg);
		
		final int altKernel = shapeK[0];
		final int largKernel = shapeK[1];
		double[] dataE = entrada.paraArrayDouble();
		double[] dataK = kernel.paraArrayDouble();
		Variavel[] dataS = saida.paraArray();
		for (int i = 0; i < alt; i++) {
			for (int j = 0; j < larg; j++) {
				double soma = 0.0;
				for (int k = 0; k < altKernel; k++) {
					for (int l = 0; l < largKernel; l++) {
						soma += dataE[(k + i) * shapeE[1] + (l + j)] * dataK[k * shapeK[1] + l];
					}
				}
				dataS[i * larg + j].set(soma);
			}
		}
	
		return saida;
	}

	/**
	 * Realiza a operação de correlação cruzada entre o tensor de entrada e o kernel.
	 * @param entrada {@code Tensor} contendo os dados de entrada.
	 * @param kernel {@code Tensor} contendo o filtro que será aplicado à entrada.
	 * @param saida {@code Tensor} de destino.
	 */
	public void correlacao2D(Tensor entrada, Tensor kernel, Tensor saida) {
		if (entrada.numDim() != 2 || kernel.numDim() != 2 || saida.numDim() != 2) {
			throw new IllegalArgumentException(
				"\nTodos os tensores devem ter duas dimensões."
			);
		}

		int[] shapeE = entrada.shape();
		int[] shapeK = kernel.shape();
		int[] shapeS = saida.shape();
		
		int altEsperada  = shapeE[0] - shapeK[0] + 1;
		int largEsperada = shapeE[1] - shapeK[1] + 1;
	
		int altSaida = shapeS[0];
		int largSaida = shapeS[1];
		if (altSaida != altEsperada || largSaida != largEsperada) {
			throw new IllegalArgumentException(
				"\nDimensão de saída esperada (" + altEsperada + ", " + largEsperada + "), mas" +
				" recebido " + saida.shapeStr()
			);
		}

		final int altKernel = shapeK[0];
		final int largKernel = shapeK[1];

		// vetorização para melhorar o desempenho
		double[] dataE = entrada.paraArrayDouble();
		double[] dataK = kernel.paraArrayDouble();
		Variavel[] dataS = saida.paraArray();

		for (int i = 0; i < altEsperada; i++) {
			for (int j = 0; j < largEsperada; j++) {
				double soma = 0.0;
				for (int k = 0; k < altKernel; k++) {
					for (int l = 0; l < largKernel; l++) {
						soma += dataE[(k + i) * shapeE[1] + (l + j)] * dataK[k * shapeK[1] + l];
					}
				}
				dataS[i * largEsperada + j].set(soma);
			}
		}

	}

	/**
	 * Realiza a operação de convolução entre o tensor de entrada e o kernel.
	 * @param entrada {@code Tensor} contendo os dados de entrada.
	 * @param kernel {@code Tensor} contendo o filtro que será aplicado à entrada.
	 * @return {@code Tensor} de destino.
	 */
	public Tensor convolucao2DFull(Tensor entrada, Tensor kernel) {
		if (entrada.numDim() != 2 || kernel.numDim() != 2) {
			throw new IllegalArgumentException(
				"\nAmbos os tensores devem ter duas dimensões."
			);

		}

		int[] shapeE = entrada.shape();
		int[] shapeK = kernel.shape();
		
		int altSaida  = shapeE[0] + shapeK[0] - 1;
		int largSaida = shapeE[1] + shapeK[1] - 1;
	
		Tensor saida = new Tensor(altSaida, largSaida);

		final int altEntrada = shapeE[0];
		final int largEntrada = shapeE[1];
		final int altKernel = shapeK[0];
		final int largKernel = shapeK[1];
		
		double[] dataE = entrada.paraArrayDouble();
		double[] dataK = kernel.paraArrayDouble();
		Variavel[] dataS = saida.paraArray();
		for (int i = 0; i < altSaida; i++) {
			for (int j = 0; j < largSaida; j++) {
				double soma = 0.0;
				for (int m = 0; m < altKernel; m++) {
					int linEntrada = i - m;
					if (linEntrada >= 0 && linEntrada < altEntrada) {
						for (int n = 0; n < largKernel; n++) {
							int colEntrada = j - n;
							if (colEntrada >= 0 && colEntrada < largEntrada) {
								soma += dataK[m * largKernel + n] * dataE[linEntrada * largEntrada + colEntrada];
							}
						}
					}
				}
				dataS[i * largSaida + j].set(soma);
			}
		}
	
		return saida;
	}

	/**
	 * Realiza a operação de convolução entre o tensor de entrada e o kernel.
	 * @param entrada {@code Tensor} contendo os dados de entrada.
	 * @param kernel {@code Tensor} contendo o filtro que será aplicado à entrada.
	 * @param saida {@code Tensor} de destino.
	 */
	public void convolucao2DFull(Tensor entrada, Tensor kernel, Tensor saida) {
		if (entrada.numDim() != 2 || kernel.numDim() != 2) {
			throw new IllegalArgumentException(
				"\nTodos os tensores devem ter duas dimensões."
			);
		}

		int[] shapeE = entrada.shape();
		int[] shapeK = kernel.shape();
		int[] shapeS = saida.shape();
		
		int altEsperada  = shapeE[0] + shapeK[0] - 1;
		int largEsperada = shapeE[1] + shapeK[1] - 1;
	
		int altSaida = shapeS[0];
		int largSaida = shapeS[1];
		if (altSaida != altEsperada || largSaida != largEsperada) {
			throw new IllegalArgumentException(
				"\nDimensão de saída esperada (" + altEsperada + ", " + largEsperada + "), mas" +
				" recebido " + saida.shapeStr()
			);
		}

		final int altEntrada = shapeE[0];
		final int largEntrada = shapeE[1];
		final int altKernel = shapeK[0];
		final int largKernel = shapeK[1];

		// vetorização para melhorar o desempenho
		double[] dataE = entrada.paraArrayDouble();
		double[] dataK = kernel.paraArrayDouble();
		Variavel[] dataS = saida.paraArray();
	
		for (int i = 0; i < altEsperada; i++) {
			for (int j = 0; j < largEsperada; j++) {
				double soma = 0.0;
				for (int m = 0; m < altKernel; m++) {
					int linEntrada = i - m;
					if (linEntrada >= 0 && linEntrada < altEntrada) {
						for (int n = 0; n < largKernel; n++) {
							int colEntrada = j - n;
							if (colEntrada >= 0 && colEntrada < largEntrada) {
								soma += dataK[m * largKernel + n] * dataE[linEntrada * largEntrada + colEntrada];
							}
						}
					}
				}
				dataS[i * largEsperada + j].set(soma);
			}
		}

	}

	/**
	 * Realiza a propagação direta através da camada convolucional.
	 * @param entrada {@code Tensor} contendo a entrada da camada.
	 * @param kernel {@code Tensor} contendos o kernel/filtros da camada.
	 * @param saida {@code Tensor} de destino do resultado.
	 */
	public void convForward(Tensor entrada, Tensor kernel, Tensor saida) {
		int[] shapeE = entrada.shape();
		int[] shapeK = kernel.shape();
		int[] shapeS = saida.shape();

		final int profEntrada = shapeK[1];
		final int altEntrada = shapeE[1];
		final int largEntrada = shapeE[2];
		final int numFiltros = shapeK[0];
		final int altKernel = shapeK[2];
		final int largKernel = shapeK[3];
		
		final int altSaida = shapeS[1];
		final int largSaida = shapeS[2];
		final int altEsperada  = altEntrada  - altKernel  + 1;
		final int largEsperada = largEntrada - largKernel + 1;
		if (altEsperada != altSaida || largEsperada != largSaida) {
			throw new IllegalArgumentException(
				"\nDimensões de saída " + saida.shapeStr() + " incompatíveis"
			);
		}

		Tensor cache = new Tensor(altSaida, largSaida);
		for (int f = 0; f < numFiltros; f++){
			for (int e = 0; e < profEntrada; e++) {
				Tensor entrada2d = entrada.slice(new int[]{e, 0, 0}, new int[]{e+1, altEntrada, largEntrada});
				entrada2d.squeeze(0);// 3d -> 2d

				Tensor kernel2D = kernel.slice(new int[]{f, e, 0, 0}, new int[]{f+1, e+1, altKernel, largKernel});
				kernel2D.squeeze(0).squeeze(0);// 4d -> 2d

				correlacao2D(entrada2d, kernel2D, cache);
				cache.unsqueeze(0);
				
				saida.slice(new int[]{f, 0, 0}, new int[]{f+1, altSaida, largSaida}).add(cache);
				cache.squeeze(0);
			}
		}
	}

	/**
	 * Realiza a propagação reversa através da camada convolucional.
	 * @param entrada {@code Tensor} contendo a entrada da camada.
	 * @param kernel {@code Tensor} contendos o kernel/filtros da camada.
	 * @param gradS {@code Tensor} contendo o gradiente em relação a saída da camada.
	 * @param gradK {@code Tensor} contendo o gradiente em relação ao kernel/filtros da camada.
	 * @param gradE {@code Tensor} contendo o gradiente em relação à entrada da camada.
	 */
	public void convBackward(Tensor entrada, Tensor kernel, Tensor gradS, Tensor gradK, Tensor gradE) {
		int[] shapeE = entrada.shape();
		int[] shapeK = kernel.shape();
		int[] shapeS = gradS.shape();

		final int filtros = shapeK[0];
		final int entradas = shapeK[1];

		final int altE = shapeE[1];
		final int largE = shapeE[2];
		final int altF = shapeK[2];
		final int largF = shapeK[3];
		final int altS = shapeS[1];
		final int largS = shapeS[2];

		for (int f = 0; f < filtros; f++) {
			for (int e = 0; e < entradas; e++) {
				// gradiente dos kernels
				Tensor entrada2D = entrada.slice(new int[]{e, 0, 0}, new int[]{e+1, altE, largE});
				entrada2D.squeeze(0);//3d -> 2d

				Tensor gradSaida2D = gradS.slice(new int[]{f, 0, 0}, new int[]{f+1, altS, largS});
				gradSaida2D.squeeze(0);//3d -> 2d

				Tensor resCorr = correlacao2D(entrada2D, gradSaida2D);
				resCorr.unsqueeze(0).unsqueeze(0);
				gradK.slice(new int[]{f, e, 0, 0}, new int[]{f+1, e+1, altF, largF}).add(resCorr);
			
				// gradientes das entradas
				Tensor kernel2D = kernel.slice(new int[]{f, e, 0, 0}, new int[]{f+1, e+1, altF, largF});
				kernel2D.squeeze(0).squeeze(0);
				Tensor resConv = convolucao2DFull(gradSaida2D, kernel2D);
				gradE.slice(new int[]{e, 0, 0}, new int[]{e+1, altE, largE}).squeeze(0).add(resConv);
			}
		}
	}

}
