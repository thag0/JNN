package jnn.camadas;

import java.util.List;
import java.util.ArrayList;
import java.util.Optional;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;

import jnn.core.backend.Backend;
import jnn.core.parallel.PoolFactory;
import jnn.core.tensor.Tensor;
import jnn.nativo.JNNNative;

/**
 * Utilitário para operações de forward e backward de camadas.
 */
public class LayerOps {

	/**
	 * Operador interno.
	 */
    Backend backend = Backend.cpu();

	/**
	 * Operador para paralelização.
	 */
	private final ForkJoinPool pool = PoolFactory.pool(Runtime.getRuntime().availableProcessors());

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

		final int canais = shapeK[1];
		final int filtros = shapeK[0];

		final int altX = shapeX[1];
		final int largX = shapeX[2];
		final int altK = shapeK[2];
		final int largK = shapeK[3];

		final int areaX = altX * largX;
		final int areaK = altK * largK;
		final int areaS = (altX - altK + 1) * (largX - largK + 1);

		final double[] dataX = entrada.array();
		final double[] dataK = kernel.array();
		final double[] dataS = saida.array();

		final int offXBase = entrada.offset();
		final int offKBase = kernel.offset();
		final int offSBase = saida.offset();

		for (int f = 0; f < filtros; f++) {
			int offS = offSBase + f * areaS;

			for (int c = 0; c < canais; c++) {
				int offX = offXBase + c * areaX;
				int offK  = offKBase + (f * canais + c) * areaK;

				backend.corr2D(
					dataX, offX,
					dataK, offK,
					dataS, offS,
					largX, altX,
					largK, altK
				);
			}
		}
				
		if (bias.isPresent()) {
			Tensor b = bias.get();
			for (int i = 0; i < filtros; i++) {
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
		final int[] shapeX = entrada.shape();
		final int[] shapeK = kernel.shape();
		final int lotes = shapeX[0];
		final int canais = shapeX[1];
		final int filtros = shapeK[0];
		final int altX = shapeX[2];
		final int largX = shapeX[3];
		
		final int atlK = shapeK[2];
		final int largK = shapeK[3];

		if (Backend.jni) {
			JNNNative.conv2dForward(
				entrada.array(), entrada.offset(),
				kernel.array(), kernel.offset(),
				bias.isPresent() ? bias.get().array() : null,
				bias.isPresent() ? bias.get().offset() : 0,
				bias.isPresent(),
				saida.array(), saida.offset(),
				lotes, canais, filtros,
				altX, largX,
				atlK, largK
			);
			return;
		}
		
		final int altS = altX - atlK + 1;
		final int largS = largX - largK + 1;
		
		final int areaX = altX * largX;
		final int areaK = atlK * largK;
		final int areaY = altS * largS;
		
		final double[] x = entrada.array();
		final double[] k = kernel.array();
		final double[] s = saida.array();
		
		final int offXBase = entrada.offset();
		final int offKBase = kernel.offset();
		final int offYBase = saida.offset();
		
		List<ForkJoinTask<?>> tarefas = new ArrayList<>(filtros);
		
		for (int f = 0; f < filtros; f++) {
			final int filtro = f;
			
			tarefas.add(pool.submit(() -> {
				int offKf = offKBase + filtro * canais * areaK;
				double biasF = bias.isPresent() ? bias.get().get(filtro) : 0.0;
				
				for (int l = 0; l < lotes; l++) {
					int offY = offYBase + (l * filtros + filtro) * areaY;	
					for (int c = 0; c < canais; c++) {
						int offX = offXBase + (l * canais + c) * areaX;
						int offK = offKf + c * areaK;
						
						backend.corr2D(
							x, offX,
							k, offK,
							s, offY,
							largX, altX,
							largK, atlK
						);
					}
					
					if (bias.isPresent()) {
						for (int i = 0; i < areaY; i++) {
							s[offY + i] += biasF;
						}
					}
				}
			}));
		}
		
		for (ForkJoinTask<?> t : tarefas) {
			t.join();
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
		final int[] shapeX = entrada.shape();
		final int[] shapeK = kernel.shape();
		final int[] shapeGS = gradS.shape();

		final int filtros  = shapeK[0];
		final int canais  = shapeK[1];

		final int altX  = shapeX[1];
		final int largX  = shapeX[2];

		final int altK = shapeK[2];
		final int largK = shapeK[3];

		final int altS = shapeGS[1];
		final int largS = shapeGS[2];

		final int areaX  = altX * largX;
		final int areaK  = altK * largK;
		final int areaGS = altS * largS;

		final double[] dataX  = entrada.array();
		final double[] dataK  = kernel.array();
		final double[] dataGS = gradS.array();
		final double[] dataGK = gradK.array();
		final double[] dataGE = gradE.array();

		final int offXBase  = entrada.offset();
		final int offKBase  = kernel.offset();
		final int offGSBase = gradS.offset();
		final int offGKBase = gradK.offset();
		final int offGEBase = gradE.offset();

		for (int f = 0; f < filtros; f++) {
			final int offGS = offGSBase + f * areaGS;
			for (int c = 0; c < canais; c++) {
				final int offK  = offKBase  + (f * canais + c) * areaK;
				final int offGE = offGEBase + c * areaX;

				backend.conv2DFull(
					dataGS, offGS,
					dataK,  offK,
					dataGE, offGE,
					largS, altS,
					largK, altK
				);
			}
		}

		if (gradB.isPresent()) {
			Tensor gb = gradB.get();

			for (int f = 0; f < filtros; f++) {
				final int offGS = offGSBase + f * areaGS;
				for (int c = 0; c < canais; c++) {
					final int offX  = offXBase  + c * areaX;
					final int offGK = offGKBase + (f * canais + c) * areaK;
	
					backend.corr2D(
						dataX,  offX,
						dataGS, offGS,
						dataGK, offGK,
						largX, altX,
						largS, altS
					);
				}

				double somaBias = 0;
				for (int i = 0; i < areaGS; i++) {
					somaBias += dataGS[offGS + i];
				}
				gb.add(somaBias, f);
			}
			
		} else {
			for (int f = 0; f < filtros; f++) {
				final int offGS = offGSBase + f * areaGS;
				for (int c = 0; c < canais; c++) {
					final int offX  = offXBase  + c * areaX;
					final int offGK = offGKBase + (f * canais + c) * areaK;
	
					backend.corr2D(
						dataX,  offX,
						dataGS, offGS,
						dataGK, offGK,
						largX, altX,
						largS, altS
					);
				}
			}
		}

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
		final int[] shapeX = entrada.shape();
		final int[] shapeK = kernel.shape();

		final int lotes = shapeX[0];
		final int canais = shapeX[1];
		final int altX = shapeX[2];
		final int largX = shapeX[3];
	
		final int filtros  = shapeK[0];
		final int altK = shapeK[2];
		final int largK = shapeK[3];
	
		if (Backend.jni) {
			JNNNative.conv2dBackward(
				entrada.array(), entrada.offset(),
				kernel.array(), kernel.offset(),
				gradS.array(), gradS.offset(),
				gradK.array(), gradK.offset(),
				gradB.isPresent() ? gradB.get().array() : null,
				gradB.isPresent() ? gradB.get().offset() : 0,
				gradB.isPresent(),
				gradE.array(), gradE.offset(),
				lotes, canais, filtros, altX, largX, altK, largK
			);

			return;
		}

		final int altS  = altX  - altK  + 1;
		final int largS = largX - largK + 1;
	
		final int areaX  = altX * largX;
		final int areaK  = altK * largK;
		final int areaGS = altS * largS;
	
		final double[] x  = entrada.array();
		final double[] k  = kernel.array();
		final double[] gs = gradS.array();
		final double[] gk = gradK.array();
		final double[] ge = gradE.array();
	
		final int offXBase  = entrada.offset();
		final int offKBase  = kernel.offset();
		final int offGSBase = gradS.offset();
		final int offGEBase = gradE.offset();
	
		List<ForkJoinTask<?>> tarefas1 = new ArrayList<>(filtros);
	
		for (int f = 0; f < filtros; f++) {
			final int filtro = f;
			
			tarefas1.add(pool.submit(() -> {
				final int offKf = offKBase + filtro * canais * areaK;
				double somaBiasLocal = 0;

				for (int l = 0; l < lotes; l++) {
					final int offGS = offGSBase + (l * filtros + filtro) * areaGS;
					for (int c = 0; c < canais; c++) {
						int offX  = offXBase + (l * canais + c) * areaX;
						int offGK = offKf + c * areaK;
	
						backend.corr2D(
							x,  offX,
							gs, offGS,
							gk, offGK,
							largX, altX,
							largS, altS
						);
					}
	
					if (gradB.isPresent()) {
						for (int i = 0; i < areaGS; i++) {
							somaBiasLocal += gs[offGS + i];
						}
					}
				}
	
				if (gradB.isPresent()) {
					gradB.get().add(somaBiasLocal, filtro);
				}
			}));
		}
	
		for (ForkJoinTask<?> t : tarefas1) {
			t.join();
		}

		// tem que ser assim porque se juntar tudo no mesmo loop por filtros
		// acaba dando race condition no gradiente de entrada

		List<ForkJoinTask<?>> tarefas2 = new ArrayList<>(lotes);

		for (int l = 0; l < lotes; l++) {
			final int lote = l;

			tarefas2.add(pool.submit(() -> {
				for (int c = 0; c < canais; c++) {
					int offGE = offGEBase + (lote * canais + c) * areaX;
					for (int f = 0; f < filtros; f++) {
						int offGS = offGSBase + (lote * filtros + f) * areaGS;
						int offK  = offKBase  + (f * canais + c) * areaK;

						backend.conv2DFull(
							gs, offGS,
							k,  offK,
							ge, offGE,
							largS, altS,
							largK, altK
						);
					}
				}
			}));
		}

		for (ForkJoinTask<?> t : tarefas2) {
			t.join();
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
		
		if (Backend.jni) {
			JNNNative.maxPool2dForward(
				entrada.array(), entrada.offset(),
				saida.array(), saida.offset(),
				canais, entrada.tamDim(1), entrada.tamDim(2),
				filtro[0], filtro[1],
				stride[0], stride[1]
			);
			
			return;
		} 

		for (int i = 0; i < canais; i++) {
			backend.maxPool2D(
				entrada.subTensor(i), 
				saida.subTensor(i), 
				filtro, 
				stride
			);
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
		if (Backend.jni) {
			JNNNative.maxPool2dForwardLotes(
				entrada.array(), entrada.offset(),
				saida.array(), saida.offset(),
				entrada.tamDim(0),
				entrada.tamDim(1),
				entrada.tamDim(2),
				entrada.tamDim(3),
				filtro[0], filtro[1],
				stride[0], stride[1]
			);
			return;
		}

		// isso aqui vai melhorar ainda
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
		if (Backend.jni) {
			JNNNative.maxPool2dBackward(
				entrada.array(), entrada.offset(),
				entrada.tamDim(0),
				entrada.tamDim(1), entrada.tamDim(2),
				grad.array(), grad.offset(), grad.tamDim(1), grad.tamDim(2),
				gradE.array(), gradE.offset(), 
				filtro[0], filtro[1],
				stride[0], stride[1]
			);

			return;
		}

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
		final int lotes = entrada.tamDim(0);
		
		if (Backend.jni) {
			JNNNative.maxPool2dBackwardLotes(
				entrada.array(),
				grad.array(),
				gradE.array(),
				lotes, entrada.tamDim(1),
				entrada.tamDim(2), entrada.tamDim(3),
				grad.tamDim(2), grad.tamDim(3),
				filtro[0], filtro[1],
				stride[0], stride[1]
			);
			return;
		}

		// isso ainda vai melhorar
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
