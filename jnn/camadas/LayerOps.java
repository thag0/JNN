package jnn.camadas;

import java.util.List;
import java.util.ArrayList;
import java.util.Optional;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;

import jnn.core.backend.Backend;
import jnn.core.parallel.PoolFactory;
import jnn.core.tensor.Tensor;

/**
 * Utilitário para operações de forward e backward de camadas.
 */
public class LayerOps {

    Backend backend = Backend.cpu();

	/**
	 * Operador para paralelização.
	 */
	private final ForkJoinPool pool = PoolFactory.pool();

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
		backend.matmul(entrada, kernel, saida);

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
		backend.matmul(gradS, kernel.transpor(), gradE);
		
		if (gradS.numDim() == 1) {//amostra única
			backend.matmul(entrada.unsqueeze(0).transpor(), gradS, gradK);
			gradB.ifPresent(gb -> gb.add(gradS));
		
		} else if (gradS.numDim() == 2) {//lote de amostras
			backend.matmul(entrada.transpor(), gradS, gradK);
			
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
		
		} else if (entrada.numDim() == 4) {
			forwardConv2DLotes(entrada, kernel, bias, saida);
		
		} else {
			throw new IllegalArgumentException(
				"\nTamanho de entrada deve ser 3 ou 4, recebido " + entrada.numDim()
			);
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
		final int[] shapeK = kernel.shape();
		final int[] shapeX = entrada.shape();

		final int profEntrada = shapeK[1];
		final int numFiltros = shapeK[0];

		int H = shapeX[1];
		int W = shapeX[2];
		int kH = shapeK[2];
		int kW = shapeK[3];

		int areaX = H * W;
		int areaK = kH * kW;
		int areaS = (H - kH + 1) * (W - kW + 1);

		double[] dataX = entrada.array();
		double[] dataK = kernel.array();
		double[] dataS = saida.array();

		int offXBase = entrada.offset();
		int offKBase = kernel.offset();
		int offSBase = saida.offset();

		for (int f = 0; f < numFiltros; f++) {
			int offS = offSBase + f * areaS;

			for (int c = 0; c < profEntrada; c++) {
				int offX = offXBase + c * areaX;
				int offK  = offKBase + (f * profEntrada + c) * areaK;

				backend.corr2D(
					dataX, offX,
					dataK, offK,
					dataS, offS,
					W, H,
					kW, kH
				);
			}
		}
				
		if (bias.isPresent()) {
			Tensor b = bias.get();
			for (int i = 0; i < numFiltros; i++) {
				saida.subTensor(i).add(b.get(i));
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
		int lotes = entrada.tamDim(0);
		final int[] shapeSaida = saida.subTensor(0).shape();

		Tensor[] saidaLocal = new Tensor[lotes];
		List<ForkJoinTask<?>> tarefas = new ArrayList<>(lotes);

		for (int i = 0; i < lotes; i++) {
			final int id = i;

			tarefas.add(pool.submit(() -> {
				Tensor tmp = new Tensor(shapeSaida);

				forwardConv2DNormal(
					entrada.subTensor(id),
					kernel,
					bias,
					tmp
				);

				saidaLocal[id] = tmp;
			}));
		}

		for (int i = 0; i < tarefas.size(); i++) {
			tarefas.get(i).join();
			saida.subTensor(i).copiar(saidaLocal[i]);
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
		
		} else if (entrada.numDim() == 4) {
			backwardConv2DLotes(entrada, kernel, gradS, gradK, gradB, gradE);
		
		} else {
			throw new IllegalArgumentException(
				"\nTamanho de entrada deve ser 3 ou 4, recebido " + entrada.numDim()
			);
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
				backend.conv2DFull(gsSaida[f], kernels[f][e], gsEntrada[e]);
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
				backend.corr2D(entradas[e], gsSaida[f], gsKernels[f][e]);	
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
		int lotes = entrada.tamDim(0);

		Tensor[] localK = new Tensor[lotes];
		Tensor[] localB = gradB.isPresent() ? new Tensor[lotes] : null;

		List<ForkJoinTask<?>> tarefas = new ArrayList<>(lotes);

		for (int i = 0; i < lotes; i++) {
			final int id = i;

			ForkJoinTask<?> tarefa = pool.submit(() -> {
				localK[id] = new Tensor(gradK.shape());
				if (gradB.isPresent()) {
					localB[id] = new Tensor(gradB.get().shape());
				}
	
				backwardConv2DNormal(
					entrada.subTensor(id),
					kernel,
					gradS.subTensor(id),
					localK[id],
					Optional.of(localB[id]),
					gradE.subTensor(id)
				);
			});

			tarefas.add(tarefa);
		}

		for (var tarefa : tarefas) tarefa.join();

		for (Tensor grad : localK) gradK.add(grad);

		if (gradB.isPresent()) {
			Tensor gb = gradB.get();
			for (Tensor grad : localB) {
				gb.add(grad);
			}
		}
	}

	/**
	 * Realiza a propagação direta através da camada MaxPool2D.
	 * @param entrada {@code Tensor} contendo a entrada da camada.
	 * @param saida {@code Tensor} contendos a saída da camada.
	 * @param filtro formato do filtro {@code (altura, largura)}
	 * @param stride formato dos strides {@code (altura, largura)}
	 */
	public void forwardMaxPool2D(Tensor entrada, Tensor saida, int[] filtro, int[] stride) {
		if (entrada.numDim() == 3) {
			forwardMaxPool2DNormal(entrada, saida, filtro, stride);
		} else {
			forwardMaxPool2DLotes(entrada, saida, filtro, stride);
		}
	}

	/**
	 * Operador interno da camada MaxPool2D tradicional
	 * @param entrada entrada.
	 * @param saida saida.
	 * @param filtro filtro.
	 * @param stride stride.
	 */
	private void forwardMaxPool2DNormal(Tensor entrada, Tensor saida, int[] filtro, int[] stride) {
		final int canais = entrada.tamDim(0);
		for (int i = 0; i < canais; i++) {
			backend.maxPool2D(entrada.subTensor(i), saida.subTensor(i), filtro, stride);
		}
	}

	/**
	 * Operador interno da camada MaxPool2D para lidar com lotes
	 * @param entrada entrada.
	 * @param saida saida.
	 * @param filtro filtro.
	 * @param stride stride.
	 */
	private void forwardMaxPool2DLotes(Tensor entrada, Tensor saida, int[] filtro, int[] stride) {
		// TODO refatorar para trabalhar com os tensores de forma mais inteligente
		// por enquanto ta assim por compatibilidade.
		final int lotes = entrada.tamDim(0);
		for (int i = 0; i < lotes; i++) {
			forwardMaxPool2DNormal(
				entrada.subTensor(i), 
				saida.subTensor(i), 
				filtro, 
				stride
			);
		}
	}

	/**
	 * Realiza a propagação reversa através da camada MaxPool2D.
	 * @param entrada {@code Tensor} contendo a entrada da camada.
	 * @param grad {@code Tensor} contendo o gradiente da saída da camada.
	 * @param gradE {@code Tensor} contendo o gradiente em relação a entrada da camada.
	 * @param filtro formato do filtro {@code (altura, largura)}
	 * @param stride formato dos strides {@code (altura, largura)}
	 */
	public void backwardMaxPool2D(Tensor entrada, Tensor grad, Tensor gradE, int[] filtro, int[] stride) {
		if (entrada.numDim() == 3) {
			backwardMaxPool2DNormal(entrada, grad, gradE, filtro, stride);
		} else {
			backwardMaxPool2DLotes(entrada, grad, gradE, filtro, stride);
		}
	}

	/**
	 * Operador interno da camada MaxPool2D tradicional.
	 * @param entrada entrada.
	 * @param grad grad.
	 * @param gradE gradE.
	 * @param filtro filtro.
	 * @param stride stride.
	 */
	private void backwardMaxPool2DNormal(Tensor entrada, Tensor grad, Tensor gradE, int[] filtro, int[] stride) {
		int[] shapeEntrada = entrada.shape();
		int[] shapeGradS   = grad.shape();

		int canais      = shapeEntrada[0];
		int altEntrada  = shapeEntrada[1];
		int largEntrada = shapeEntrada[2];

		int altGradS    = shapeGradS[1];
		int largGradS   = shapeGradS[2];

		double[] dataE  = entrada.array();
		double[] dataGS = grad.array();
		double[] dataGE = gradE.array();

		int canalSizeEntrada = altEntrada * largEntrada;
		int canalSizeGradS   = altGradS * largGradS;
		double val, valMax;

		for (int c = 0; c < canais; c++) {
			int baseEntrada = c * canalSizeEntrada;
			int baseGradS   = c * canalSizeGradS;

			for (int i = 0; i < altGradS; i++) {
				int linInicio = i * stride[0];
				int linFim    = Math.min(linInicio + filtro[0], altEntrada);

				for (int j = 0; j < largGradS; j++) {
					int colInicio = j * stride[1];
					int colFim    = Math.min(colInicio + filtro[1], largEntrada);

					valMax = Double.NEGATIVE_INFINITY;
					int linMax = linInicio;
					int colMax = colInicio;

					// Encontrar posição do máximo
					for (int y = linInicio; y < linFim; y++) {
						int idLinha = baseEntrada + y * largEntrada;
						for (int x = colInicio; x < colFim; x++) {
							val = dataE[idLinha + x];
							if (val > valMax) {
								valMax = val;
								linMax = y;
								colMax = x;
							}
						}
					}

					dataGE[baseEntrada + linMax * largEntrada + colMax] += dataGS[baseGradS + i * largGradS + j];
				}
			}
		}
	}

	/**
	 * Operador interno da camada MaxPool2D para lidar com lotes.
	 * @param entrada entrada.
	 * @param grad grad.
	 * @param gradE gradE.
	 * @param filtro filtro.
	 * @param stride stride.
	 */
	private void backwardMaxPool2DLotes(Tensor entrada, Tensor grad, Tensor gradE, int[] filtro, int[] stride) {
		// TODO melhorar isso
		// por enquanto vai ficar assim por compatibilidade.

		final int lotes = entrada.tamDim(0);
		for (int i = 0; i < lotes; i++) {
			backwardMaxPool2DNormal(
				entrada.subTensor(i), 
				grad.subTensor(i), 
				gradE.subTensor(i), 
				filtro, 
				stride
			);
		}
	}

	/**
	 * Realiza a propagação direta através da camada MaxPool2D.
	 * @param entrada {@code Tensor} contendo a entrada da camada.
	 * @param saida {@code Tensor} contendos a saída da camada.
	 * @param filtro formato do filtro {@code (altura, largura)}
	 * @param stride formato dos strides {@code (altura, largura)}
	 */
	public void forwardAvgPool2D(Tensor entrada, Tensor saida, int[] filtro, int[] stride) {
		if (entrada.numDim() == 3) {
			forwardAvgPool2DNormal(entrada, saida, filtro, stride);
		} else {
			forwardAvgPool2DLotes(entrada, saida, filtro, stride);
		}
	}

	/**
	 * Operador interno da camada AvgPool2D tradicional.
	 * @param entrada {@code Tensor} contendo a entrada da camada.
	 * @param saida {@code Tensor} contendos a saída da camada.
	 * @param filtro formato do filtro {@code (altura, largura)}
	 * @param stride formato dos strides {@code (altura, largura)}
	 */
	private void forwardAvgPool2DNormal(Tensor entrada, Tensor saida, int[] filtro, int[] stride) {
		final int canais = entrada.tamDim(0);
		for (int i = 0; i < canais; i++) {
			backend.avgPool2D(entrada.subTensor(i), saida.subTensor(i), filtro, stride);
		}
	}

	/**
	 * Operador interno da camada AvgPool2D para lidar com lotes.
	 * @param entrada {@code Tensor} contendo a entrada da camada.
	 * @param saida {@code Tensor} contendos a saída da camada.
	 * @param filtro formato do filtro {@code (altura, largura)}
	 * @param stride formato dos strides {@code (altura, largura)}
	 */
	private void forwardAvgPool2DLotes(Tensor entrada, Tensor saida, int[] filtro, int[] stride) {
		// TODO refatorar para trabalhar com os tensores de forma mais inteligente
		// por enquanto ta assim por compatibilidade.
		final int lotes = entrada.tamDim(0);
		for (int i = 0; i < lotes; i++) {
			forwardAvgPool2DNormal(
				entrada.subTensor(i), 
				saida.subTensor(i), 
				filtro, 
				stride
			);
		}
	}

	/**
	 * Realiza a propagação reversa através da camada MaxPool2D.
	 * @param entrada {@code Tensor} contendo a entrada da camada.
	 * @param grad {@code Tensor} contendo o gradiente da saída da camada.
	 * @param gradE {@code Tensor} contendo o gradiente em relação a entrada da camada.
	 * @param filtro formato do filtro {@code (altura, largura)}
	 * @param stride formato dos strides {@code (altura, largura)}
	 */
	public void backwardAvgPool(Tensor entrada, Tensor grad, Tensor gradE, int[] filtro, int[] stride) {
		if (entrada.numDim() == 3) {
			backwardAvgPool2DNormal(entrada, grad, gradE, filtro, stride);
		} else {
			backwardAvgPool2DLotes(entrada, grad, gradE, filtro, stride);
		}
	}

	/**
	 * Operador interno da camada AvgPool2D tradicional.
	 * @param entrada {@code Tensor} contendo a entrada da camada.
	 * @param grad {@code Tensor} contendo o gradiente da saída da camada.
	 * @param gradE {@code Tensor} contendo o gradiente em relação a entrada da camada.
	 * @param filtro formato do filtro {@code (altura, largura)}
	 * @param stride formato dos strides {@code (altura, largura)}
	 */
	private void backwardAvgPool2DNormal(Tensor entrada, Tensor grad, Tensor gradE, int[] filtro, int[] stride) {
		int C = entrada.shape()[0];
		int Hin  = entrada.shape()[1];
		int Win  = entrada.shape()[2];
		int Hout = grad.shape()[1];
		int Wout = grad.shape()[2];

		int fH = filtro[0];
		int fW = filtro[1];
		int sH = stride[0];
		int sW = stride[1];

		double[] arrGo  = grad.array();
		double[] arrGi  = gradE.array();

		int offGo  = grad.offset();
		int offGi  = gradE.offset();

		int[] stGo  = grad.strides();
		int[] stGi  = gradE.strides();

		int janela = fH * fW;

		for (int c = 0; c < C; c++) {
			int baseGo  = offGo + c * stGo[0];
			int baseGi  = offGi + c * stGi[0];

			for (int i = 0; i < Hout; i++) {
				int linInicio = i * sH;
				int linFim = Math.min(linInicio + fH, Hin);

				for (int j = 0; j < Wout; j++) {
					int colInicio = j * sW;
					int colFim = Math.min(colInicio + fW, Win);
					double g = arrGo[baseGo + i * stGo[1] + j * stGo[2]] / janela;

					for (int y = linInicio; y < linFim; y++) {
						int linhaGi = baseGi + y * stGi[1];
						for (int x = colInicio; x < colFim; x++) {
							arrGi[linhaGi + x * stGi[2]] += g;
						}
					}
				}
			}
		}
	}

	/**
	 * Operador interno da camada AvgPool2D para lidar com lotes.
	 * @param entrada {@code Tensor} contendo a entrada da camada.
	 * @param grad {@code Tensor} contendo o gradiente da saída da camada.
	 * @param gradE {@code Tensor} contendo o gradiente em relação a entrada da camada.
	 * @param filtro formato do filtro {@code (altura, largura)}
	 * @param stride formato dos strides {@code (altura, largura)}
	 */
	private void backwardAvgPool2DLotes(Tensor entrada, Tensor grad, Tensor gradE, int[] filtro, int[] stride) {
		// TODO melhorar isso
		// por enquanto vai ficar assim por compatibilidade.
		final int lotes = entrada.tamDim(0);
		for (int i = 0; i < lotes; i++) {
			backwardAvgPool2DNormal(
				entrada.subTensor(i), 
				grad.subTensor(i), 
				gradE.subTensor(i), 
				filtro, 
				stride
			);
		}
	}

}
