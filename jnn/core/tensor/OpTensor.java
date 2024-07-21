package jnn.core.tensor;

import java.util.Optional;

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
		Variavel[] dataA = a.paraArray();
		Variavel[] dataB = b.paraArray();
		Variavel[] dataD = dest.paraArray();
		
		// cache
		final int n = colA;
		Variavel soma = new Variavel();

		for (int i = 0; i < linD; i++) {
			for (int j = 0; j < colD; j++) {
				soma.set(0.0);
				int idSaida = (i * colD) + j;
				for (int k = 0; k < n; k++) {
					soma.addMult(
						dataA[i * colA + k],
						dataB[k * colB + j]
					);
				}
				dataD[idSaida].set(soma);
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
		
		int altEsp  = shapeE[0] - shapeK[0] + 1;
		int largEsp = shapeE[1] - shapeK[1] + 1;
	
		int altSaida = shapeS[0];
		int largSaida = shapeS[1];
		if (altSaida != altEsp || largSaida != largEsp) {
			throw new IllegalArgumentException(
				"\nDimensão de saída esperada (" + altEsp + ", " + largEsp + "), mas" +
				" recebido " + saida.shapeStr()
			);
		}

		final int altKernel = shapeK[0];
		final int largKernel = shapeK[1];
		final int largEntrada = shapeE[1];

		// vetorização para melhorar o desempenho
		Variavel[] dataE = entrada.paraArray();
		Variavel[] dataK = kernel.paraArray();
		Variavel[] dataS = saida.paraArray();

		Variavel soma = new Variavel();
		for (int i = 0; i < altEsp; i++) {
			for (int j = 0; j < largEsp; j++) {

				soma.set(0.0);
				final int idSaida = i * largEsp + j;
				for (int k = 0; k < altKernel; k++) {
					final int idBaseEntrada = (k + i) * largEntrada;
					final int idBaseKernel  = k * largKernel; 
					for (int l = 0; l < largKernel; l++) {
						soma.addMult(
							dataE[idBaseEntrada + (l + j)],
							dataK[idBaseKernel + l]
						);
					}
				}

				dataS[idSaida].add(soma);
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
		
		int altEsp  = shapeE[0] - shapeK[0] + 1;
		int largEsp = shapeE[1] - shapeK[1] + 1;
	
		int altSaida = shapeS[0];
		int largSaida = shapeS[1];
		if (altSaida != altEsp || largSaida != largEsp) {
			throw new IllegalArgumentException(
				"\nDimensão de saída esperada (" + altEsp + ", " + largEsp + "), mas" +
				" recebido " + saida.shapeStr()
			);
		}
	
		final int altKernel = shapeK[0];
		final int largKernel = shapeK[1];
		final int largEntrada = shapeE[1];
	
		// vetorização para melhorar o desempenho
		Variavel[] dataE = entrada.paraArray();
		Variavel[] dataK = kernel.paraArray();
		Variavel[] dataS = saida.paraArray();
	
		Variavel soma = new Variavel();
		for (int i = 0; i < altEsp; i++) {
			for (int j = 0; j < largEsp; j++) {
				
				soma.set(0.0);
				final int idSaida = i * largEsp + j;
				for (int k = 0; k < altKernel; k++) {
					for (int l = 0; l < largKernel; l++) {
						soma.addMult(
							dataE[(k + i) * largEntrada + (l + j)], 
							dataK[(altKernel - 1 - k) * largKernel + (largKernel - 1 - l)]
						);
					}
				}

				dataS[idSaida].add(soma);
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

		final int altEntrada = shapeE[0];
		final int largEntrada = shapeE[1];
		final int altKernel = shapeK[0];
		final int largKernel = shapeK[1];

		// vetorização para melhorar o desempenho
		Variavel[] dataE = entrada.paraArray();
		Variavel[] dataK = kernel.paraArray();
		Variavel[] dataS = saida.paraArray();

		Variavel soma = new Variavel();// cache
		for (int i = 0; i < altEsp; i++) {
			for (int j = 0; j < largEsp; j++) {
				
				soma.set(0.0);
				final int idSaida = i*largEsp + j;
				for (int m = 0; m < altKernel; m++) {
					int linEntrada = i - m;
					if (linEntrada >= 0 && linEntrada < altEntrada) {
						for (int n = 0; n < largKernel; n++) {
							int colEntrada = j - n;
							if (colEntrada >= 0 && colEntrada < largEntrada) {
								soma.addMult(
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
	 * Realiza a peopagação direta através da camada densa.
	 * @param entrada {@code Tensor} contendo a entrada da camada.
	 * @param kernel {@code Tensor} contendos o kernel/pesos da camada.
	 * @param bias {@code Tensor} contendo o bias da camada {@code (podendo ser nulo)}.
	 * @param saida {@code Tensor} de destino do resultado.
	 */
	public void densaForward(Tensor entrada, Tensor kernel, Optional<Tensor> bias, Tensor saida) {
		matMult(entrada, kernel, saida);
		bias.ifPresent(b -> saida.add(b));
	}

	/**
	 * Realiza a propagação reversa através da camada densa.
	 * @param entrada {@code Tensor} contendo a entrada da camada.
	 * @param kernel {@code Tensor} contendos o kernel/pesos da camada.
	 * @param gradS {@code Tensor} contendo o gradiente em relação a saída da camada.
	 * @param gradK {@code Tensor} contendo o gradiente em relação ao kernel/pesos da camada.
	 * @param gradB {@code Tensor} contendo o gradiente em relação ao bias da camada {@code (podendo ser nulo)}.
	 * @param gradE {@code Tensor} contendo o gradiente em relação à entrada da camada.
	 */
	public void densaBackward(Tensor entrada, Tensor kernel, Tensor gradS, Tensor gradK, Optional<Tensor> gradB, Tensor gradE) {
		Tensor temp = new Tensor(gradK.shape());
		matMult(entrada.transpor(), gradS, temp);
		gradK.add(temp);

		gradB.ifPresent(gb -> gb.add(gradS));

		matMult(gradS, kernel.transpor(), gradE);
	}

	/**
	 * Realiza a propagação direta através da camada convolucional.
	 * @param entrada {@code Tensor} contendo a entrada da camada.
	 * @param kernel {@code Tensor} contendos o kernel/filtros da camada.
	 * @param bias {@code Tensor} contendo o bias da camada {@code (podendo ser nulo)}.
	 * @param saida {@code Tensor} de destino do resultado.
	 */
	public void conv2DForward(Tensor entrada, Tensor kernel, Optional<Tensor> bias, Tensor saida) {
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
		// essa ainda não é a melhor solução, mas é mais eficiente que fazer 
		// slicing dentro dos loops.
		Tensor entrada2D = new Tensor(altEntrada, largEntrada);
		Tensor kernel2D = new Tensor(altKernel, largKernel);
		Tensor cache = new Tensor(altSaida, largSaida);
		
		for (int f = 0; f < numFiltros; f++){
			cache.zerar();// zerar acumulações para o filtro atual

			for (int e = 0; e < profEntrada; e++) {
				for (int i = 0; i < altEntrada; i++) {
					for (int j = 0; j < largEntrada; j++) {
						entrada2D.set(entrada.get(e, i, j), i, j);
					}
				}

				for (int i = 0; i < altKernel; i++) {
					for (int j = 0; j < largKernel; j++) {
						kernel2D.set(kernel.get(f, e, i, j), i, j);
					}
				}

				correlacao2D(entrada2D, kernel2D, cache);
			}

			for (int i = 0; i < altSaida; i++) {
				for (int j = 0; j < largSaida; j++) {
					saida.add(cache.get(i, j), f, i, j);
				}
			}
		}

		bias.ifPresent(b -> {
			for (int i = 0; i < numFiltros; i++) {
				double val = b.get(i);
				for (int j = 0; j < altSaida; j++) {
					for (int k = 0; k < largSaida; k++) {
						saida.add(val, i, j, k);
					}
				}
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
	 */
	public void conv2DBackward(Tensor entrada, Tensor kernel, Tensor gradS, Tensor gradK, Optional<Tensor> gradB, Tensor gradE) {
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

		boolean temBias = gradB.isPresent();

		// NOTA
		// essa ainda não é a melhor solução, mas é mais eficiente que 
		// fazer slicing dentro dos loops.

		// aproveitar paralelismo para dividir o trabalho e sobrecarregar
		// menos um único núcleo do processador.

		// gradiente em relação as entradas
		Thread t1 = new Thread(() -> {
			Tensor kernel2D = new Tensor(altK, largK);
			Tensor gradSaida2D = new Tensor(altS, largS);
			Tensor cache = new Tensor(altE, largE);

			for (int e = 0; e < entradas; e++) {
				cache.zerar();// zerar acumulador
				for (int f = 0; f < filtros; f++) {

					for (int i = 0; i < altK; i++) {
						for (int j = 0; j < largK; j++) {
							kernel2D.set(kernel.get(f, e, i, j), i, j);
						}
					}
	
					for (int i = 0; i < altS; i++) {
						for (int j = 0; j < largS; j++) {
							gradSaida2D.set(gradS.get(f, i, j), i, j);
						}
					}
	
					convolucao2DFull(gradSaida2D, kernel2D, cache);
				}

				for (int i = 0; i < altE; i++) {
					for (int j = 0; j < largE; j++) {
						gradE.add(cache.get(i, j), e, i, j);
					}
				}
			}
		});
		t1.start();

		// gradiente em relação aos kernels
		Thread t2 = new Thread(() -> {
			Tensor entrada2D = new Tensor(altE, largE);
			Tensor gradSaida2D = new Tensor(altS, largS);
			Tensor cache = new Tensor(altK, largK);

			for (int f = 0; f < filtros; f++) {
				for (int e = 0; e < entradas; e++) {
					cache.zerar();	
					for (int i = 0; i < altE; i++) {
						for (int j = 0; j < largE; j++) {
							entrada2D.set(entrada.get(e, i, j), i, j);
						}
					}
	
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
		});
		t2.start();

		// gradiente em relação aos bias
		Thread t3 = null;
		if (temBias) {
			t3 = new Thread(() -> {
				for (int i = 0; i < filtros; i++) {
					double soma = 0.0;
					for (int j = 0; j < altS; j++) {
						for (int k = 0; k < largS; k++) {
							soma += gradS.get(i, j, k);
						}
					}
					gradB.get().add(soma, i);
				}
			});
			t3.start();
		}
	
		try {
			t1.join();
			t2.join();
			if (temBias) t3.join();
		} catch (InterruptedException e) {
			System.out.println(e.getMessage());
		}
	}

}
