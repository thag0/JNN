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
		
		matMult(a, b, res);
	
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
	
		int[] shapeA = a.shape();
		int[] shapeB = b.shape();
		int[] shapeD = dest.shape();

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

		correlacao2D(entrada, kernel, saida);

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
		final int largEntrada = shapeE[1];

		// vetorização para melhorar o desempenho
		double[] dataE = entrada.paraArrayDouble();
		double[] dataK = kernel.paraArrayDouble();
		Variavel[] dataS = saida.paraArray();

		for (int i = 0; i < altEsperada; i++) {
			for (int j = 0; j < largEsperada; j++) {
				final int idSaida = i * largEsperada + j;
				double soma = 0.0;
				for (int k = 0; k < altKernel; k++) {
					for (int l = 0; l < largKernel; l++) {
						soma += dataE[(k + i) * largEntrada + (l + j)] * dataK[k * largKernel + l];
					}
				}
				dataS[idSaida].set(soma);
			}
		}

	}

	/**
	 * Realiza a operação de convolução entre o tensor de entrada e o kernel.
	 * @param entrada {@code Tensor} contendo os dados de entrada.
	 * @param kernel {@code Tensor} contendo o filtro que será aplicado à entrada.
	 * @return {@code Tensor} de destino.
	 */
	public Tensor convolucao2D(Tensor entrada, Tensor kernel) {
		if (entrada.numDim() != 2 || kernel.numDim() != 2) {
			throw new IllegalArgumentException(
				"\nTodos os tensores devem ter duas dimensões."
			);
		}

		int[] shapeE = entrada.shape();
		int[] shapeK = kernel.shape();
		
		int alt  = shapeE[0] - shapeK[0] + 1;
		int larg = shapeE[1] - shapeK[1] + 1;
		Tensor res = new Tensor(alt, larg);

		convolucao2D(entrada, kernel, res);

		return res;
	}

	/**
	 * Realiza a operação de convolução entre o tensor de entrada e o kernel.
	 * @param entrada {@code Tensor} contendo os dados de entrada.
	 * @param kernel {@code Tensor} contendo o filtro que será aplicado à entrada.
	 * @param saida {@code Tensor} de destino.
	 */
	public void convolucao2D(Tensor entrada, Tensor kernel, Tensor saida) {
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
		final int largEntrada = shapeE[1];
	
		// vetorização para melhorar o desempenho
		double[] dataE = entrada.paraArrayDouble();
		double[] dataK = kernel.paraArrayDouble();
		Variavel[] dataS = saida.paraArray();
	
		for (int i = 0; i < altEsperada; i++) {
			for (int j = 0; j < largEsperada; j++) {
				final int idSaida = i * largEsperada + j;
				double soma = 0.0;
				for (int k = 0; k < altKernel; k++) {
					for (int l = 0; l < largKernel; l++) {
						soma += 
						dataE[(k + i) * largEntrada + (l + j)] * 
						dataK[(altKernel - 1 - k) * largKernel + (largKernel - 1 - l)];
					}
				}
				dataS[idSaida].set(soma);
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
		
		int alt  = shapeE[0] + shapeK[0] - 1;
		int larg = shapeE[1] + shapeK[1] - 1;
	
		Tensor saida = new Tensor(alt, larg);

		convolucao2DFull(entrada, kernel, saida);
	
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
	public void conv2DForward(Tensor entrada, Tensor kernel, Tensor saida) {
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

		// NOTA
		// essa ainda não é a melhor solução, mas é mais eficiente que fazer slicing
		// dentro dos loops.
		Tensor cache = new Tensor(altSaida, largSaida);
		for (int f = 0; f < numFiltros; f++){
			for (int e = 0; e < profEntrada; e++) {
				Tensor entrada2d = new Tensor(altEntrada, largEntrada);
				for (int i = 0; i < altEntrada; i++) {
					for (int j = 0; j < largEntrada; j++) {
						entrada2d.set(entrada.get(e, i, j), i, j);
					}
				}

				Tensor kernel2D = new Tensor(altKernel, largKernel);
				for (int i = 0; i < altKernel; i++) {
					for (int j = 0; j < largKernel; j++) {
						kernel2D.set(kernel.get(f, e, i, j), i, j);
					}
				}

				correlacao2D(entrada2d, kernel2D, cache);

				for (int i = 0; i < altSaida; i++) {
					for (int j = 0; j < largSaida; j++) {
						saida.add(cache.get(i, j), f, i, j);
					}
				}
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
	public void conv2DBackward(Tensor entrada, Tensor kernel, Tensor gradS, Tensor gradK, Tensor gradE) {
		int[] shapeE = entrada.shape();
		int[] shapeK = kernel.shape();
		int[] shapeS = gradS.shape();

		final int filtros = shapeK[0];
		final int entradas = shapeK[1];

		final int altE = shapeE[1];
		final int largE = shapeE[2];
		final int altK = shapeK[2];
		final int largK = shapeK[3];
		final int altS = shapeS[1];
		final int largS = shapeS[2];

		// NOTA
		// essa ainda não é a melhor solução, mas é mais eficiente que 
		// fazer slicing dentro dos loops.

		// gradientes das entradas
		Thread t = new Thread(() -> {
			Tensor cache = new Tensor(altE, largE);
			for (int f = 0; f < filtros; f++) {
				for (int e = 0; e < entradas; e++) {
					Tensor kernel2D = new Tensor(altK, largK);
					for (int i = 0; i < altK; i++) {
						for (int j = 0; j < largK; j++) {
							kernel2D.set(kernel.get(f, e, i, j), i, j);
						}
					}
	
					Tensor gradSaida2D = new Tensor(altS, largS);
					for (int i = 0; i < altS; i++) {
						for (int j = 0; j < largS; j++) {
							gradSaida2D.set(gradS.get(f, i, j), i, j);
						}
					}
	
					convolucao2DFull(gradSaida2D, kernel2D, cache);
					
					for (int i = 0; i < altE; i++) {
						for (int j = 0; j < largE; j++) {
							gradE.add(cache.get(i, j), e, i, j);
						}
					}
				}
			}
		});
		t.start();

		// gradiente dos kernels
		Tensor cache = new Tensor(altK, largK);
		for (int f = 0; f < filtros; f++) {
			for (int e = 0; e < entradas; e++) {
				Tensor entrada2D = new Tensor(altE, largE);
				for (int i = 0; i < altE; i++) {
					for (int j = 0; j < largE; j++) {
						entrada2D.set(entrada.get(e, i, j), i, j);
					}
				}

				Tensor gradSaida2D = new Tensor(altS, largS);
				for (int i = 0; i < altS; i++) {
					for (int j = 0; j < largS; j++) {
						gradSaida2D.set(gradS.get(f, i, j), i, j);
					}
				}

				correlacao2D(entrada2D, gradSaida2D, cache);	
				
				for (int i = 0; i < altK; i++) {
					for (int j = 0; j < largK; j++) {
						gradK.add(cache.get(i, j), f, e, i, j);
					}
				}
			}
		}
	
		try {
			t.join();
		} catch (InterruptedException e) {
			System.out.println(e.getMessage());
		}
	}

}
