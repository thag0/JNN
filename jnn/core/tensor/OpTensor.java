package jnn.core.tensor;

import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

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
	public Tensor matMul(Tensor a, Tensor b) {
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
		
		matMul(a, b, res);
	
		return res;
	}

	/**
	 * Realiza a operação {@code  A * B}
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @param dest {@code Tensor} de destino.
	 */
	public void matMul(Tensor a, Tensor b, Tensor dest) {
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
				soma.zero();
				int idSaida = (i * colD) + j;
				for (int k = 0; k < n; k++) {
					soma.addMul(
						dataA[i * colA + k],
						dataB[k * colB + j]
					);
				}
				dataD[idSaida].add(soma);
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

				soma.zero();
				final int idSaida = i * largEsp + j;
				for (int k = 0; k < altKernel; k++) {
					final int idBaseEntrada = (k + i) * largEntrada;
					final int idBaseKernel  = k * largKernel; 
					for (int l = 0; l < largKernel; l++) {
						soma.addMul(
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
				
				soma.zero();
				final int idSaida = i * largEsp + j;
				for (int k = 0; k < altKernel; k++) {
					for (int l = 0; l < largKernel; l++) {
						soma.addMul(
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
				
				soma.zero();
				final int idSaida = i*largEsp + j;
				for (int m = 0; m < altKernel; m++) {
					int linEntrada = i - m;
					if (linEntrada >= 0 && linEntrada < altEntrada) {
						for (int n = 0; n < largKernel; n++) {
							int colEntrada = j - n;
							if (colEntrada >= 0 && colEntrada < largEntrada) {
								soma.addMul(
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
		matMul(entrada, kernel, saida);
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
		matMul(entrada.transpor(), gradS, gradK);
		gradB.ifPresent(gb -> gb.add(gradS));
		matMul(gradS, kernel.transpor(), gradE);
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

		final int numThreads = 2;// diminuir sobrecarga numa única cpu
		try (ExecutorService exec = Executors.newFixedThreadPool(numThreads)) {
			for (int f = 0; f < numFiltros; f++){
				for (int e = 0; e < profEntrada; e++) {
					correlacao2D(entradas[e], kernels[f][e], saidas[f]);
				}
			}
		}

		bias.ifPresent(b -> {
			for (int i = 0; i < numFiltros; i++) {
				saida.subTensor(i).add(b.get(i));
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
		int[] shapeK = kernel.shape();

		final int numFiltros = shapeK[0];
		final int profEntrada = shapeK[1];
		boolean temBias = gradB.isPresent();

		// NOTA
		// essa ainda não é a melhor solução, mas é mais eficiente que 
		// fazer slicing dentro dos loops.

		// aproveitar paralelismo para dividir o trabalho e sobrecarregar
		// menos um único núcleo do processador.

		Tensor[] gsSaida = new Tensor[numFiltros];
		for (int i = 0; i < numFiltros; i++) {
			gsSaida[i] = gradS.subTensor(i);
		}

		// gradiente em relação as entradas
		Thread t1 = new Thread(() -> {
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
					convolucao2DFull(gsSaida[f], kernels[f][e], gsEntrada[e]);
				}
			}
		});
		t1.start();

		// gradiente em relação aos kernels
		Thread t2 = new Thread(() -> {
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
					correlacao2D(entradas[e], gsSaida[f], gsKernels[f][e]);	
				}
			}
		});
		t2.start();

		// gradiente em relação aos bias
		Thread t3 = null;
		if (temBias) {
			t3 = new Thread(() -> {
				for (int i = 0; i < numFiltros; i++) {
					double soma = gradS.subTensor(i).soma().item();
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
