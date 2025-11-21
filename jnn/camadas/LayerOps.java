package jnn.camadas;

import java.util.Optional;

import jnn.core.OpTensor;
import jnn.core.tensor.Tensor;

/**
 * Utilitário para operações de forward e backward de camadas.
 */
public class LayerOps {

    OpTensor opt = new OpTensor();

    /**
     * Utilitário para operações de forward e backward de camadas.
     */
    public LayerOps() {}
    
	/**
	 * Realiza a peopagação direta através da camada Densa.
	 * @param entrada {@code Tensor} contendo a entrada da camada.
	 * @param kernel {@code Tensor} contendos o kernel/pesos da camada.
	 * @param bias {@code Tensor} contendo o bias da camada {@code (podendo ser nulo)}.
	 * @param saida {@code Tensor} de destino do resultado.
	 * @see {@link jnn.camadas.Densa}
	 */
	public void forwardDensa(Tensor entrada, Tensor kernel, Optional<Tensor> bias, Tensor saida) {
		opt.matmul(entrada, kernel, saida);

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
		opt.matmul(gradS, kernel.transpor(), gradE);
		
		if (gradS.numDim() == 1) {//amostra única
			opt.matmul(entrada.unsqueeze(0).transpor(), gradS, gradK);
			gradB.ifPresent(gb -> gb.add(gradS));
		
		} else if (gradS.numDim() == 2) {//lote de amostras
			opt.matmul(entrada.transpor(), gradS, gradK);
			
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
		if (entrada.numDim() == 3) {
			forwardConv2DNormal(entrada, kernel, bias, saida);	
		} else {
			forwardConv2DLotes(entrada, kernel, bias, saida);
		}
	}

	/**
	 * Operador interno da camada Conv2D tradicional.
	 * @param entrada entrada.
	 * @param kernel kernel.
	 * @param bias bias.
	 * @param saida saída.
	 */
	private void forwardConv2DNormal(Tensor entrada, Tensor kernel, Optional<Tensor> bias, Tensor saida) {
		int[] shapeK = kernel.shape();
		final int profEntrada = shapeK[1];
		final int numFiltros = shapeK[0];

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
					opt.corr2D(entradas[e], kernels[i][e], saidas[i]);
				}
				saidas[i].add(b.get(i));
			}

		} else {
			for (int i = 0; i < numFiltros; i++) {
				for (int e = 0; e < profEntrada; e++) {
					opt.corr2D(entradas[e], kernels[i][e], saidas[i]);
				}
			}
		}
	}

	/**
	 * Operador interno da camada Conv2D para lidar com lotes.
	 * @param entrada entrada.
	 * @param kernel kernel.
	 * @param bias bias.
	 * @param saida saída.
	 */
	private void forwardConv2DLotes(Tensor entrada, Tensor kernel, Optional<Tensor> bias, Tensor saida) {
		// TODO refatorar para trabalhar com os tensores de forma mais inteligente
		// por enquanto ta assim por compatibilidade.
		int lotes = entrada.tamDim(0);
		for (int i = 0; i < lotes; i++) {
			forwardConv2DNormal(
				entrada.subTensor(i),
				kernel,
				bias,
				saida.subTensor(i)
			);
		}
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
		if (entrada.numDim() == 3) {
			backwardConv2DNormal(entrada, kernel, gradS, gradK, gradB, gradE);
		} else {
			backwardConv2DLotes(entrada, kernel, gradS, gradK, gradB, gradE);
		}
	}

	/**
	 * Operador interno da camada Conv2D tradicional.
	 * @param entrada entrada
	 * @param kernel kernel
	 * @param gradS gradS
	 * @param gradK gradK
	 * @param gradB gradB
	 * @param gradE gradE
	 */
	private void backwardConv2DNormal(Tensor entrada, Tensor kernel, Tensor gradS, Tensor gradK, Optional<Tensor> gradB, Tensor gradE) {
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
				opt.conv2DFull(gsSaida[f], kernels[f][e], gsEntrada[e]);
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
				opt.corr2D(entradas[e], gsSaida[f], gsKernels[f][e]);	
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

	/**
	 * Operador interno da camada Conv2D para lidar com lotes.
	 * @param entrada entrada
	 * @param kernel kernel
	 * @param gradS gradS
	 * @param gradK gradK
	 * @param gradB gradB
	 * @param gradE gradE
	 */
	private void backwardConv2DLotes(Tensor entrada, Tensor kernel, Tensor gradS, Tensor gradK, Optional<Tensor> gradB, Tensor gradE) {
		// TODO refatorar para trabalhar com os tensores de forma mais inteligente
		// por enquanto ta assim por compatibilidade.
		int lotes = entrada.tamDim(0);
		for (int i = 0; i < lotes; i++) {
			backwardConv2DNormal(
				entrada.subTensor(i),
				kernel,
				gradS.subTensor(i),
				gradK,
				gradB,
				gradE.subTensor(i)
			);
		}		
	}

}
