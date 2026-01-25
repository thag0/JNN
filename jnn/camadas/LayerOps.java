package jnn.camadas;

import java.util.List;
import java.util.ArrayList;
import java.util.Optional;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;

import jnn.core.ops.Ops;
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
    Ops ops = Ops.get();

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
		saida.zero();// zerar acumulos anteriores

		ops.matmul(entrada, kernel, saida);

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
		ops.matmul(gradS, kernel.transpor(), gradE);
		
		if (gradS.numDim() == 1) {//amostra única
			ops.matmul(entrada.unsqueeze(0).transpor(), gradS, gradK);
			gradB.ifPresent(gb -> gb.add(gradS));
		
		} else if (gradS.numDim() == 2) {//lote de amostras
			ops.matmul(entrada.transpor(), gradS, gradK);
			
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
		saida.zero();// zerar acumulos anteriores

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

				ops.corr2D(
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
		
		final int altK = shapeK[2];
		final int largK = shapeK[3];

		if (JNNNative.jni) {
			JNNNative.conv2dForward(
				entrada.array(),
				kernel.array(),
				bias.isPresent() ? bias.get().array() : null,
				bias.isPresent(),
				saida.array(),
				lotes, canais, filtros,
				altX, largX,
				altK, largK
			);

			return;
		}

		final int altS = altX - altK + 1;
		final int largS = largX - largK + 1;

		final int areaX = altX * largX;
		final int areaK = altK * largK;
		final int areaS = altS * largS;

		final double[] dataX = entrada.array();
		final double[] dataK = kernel.array();
		final double[] dataS = saida.array();
		final double[] dataB = bias.isPresent() ? bias.get().array() : null;

		final int offXBase = entrada.offset();
		final int offKBase = kernel.offset();
		final int offYBase = saida.offset();

		List<ForkJoinTask<?>> tarefas = new ArrayList<>(filtros);
		
		for (int f = 0; f < filtros; f++) {
			final int filtro = f;
			
			tarefas.add(pool.submit(() -> {
				final int offKf = offKBase + filtro * canais * areaK;
				final double biasF = (dataB != null) ? dataB[filtro] : 0.0f;
				
				for (int l = 0; l < lotes; l++) {
					final int offY = offYBase + (l * filtros + filtro) * areaS;
					final int offX_l = offXBase + (l * canais) * areaX;

					for (int i = 0; i < areaS; i++) {
						dataS[offY + i] = biasF;
					}

					for (int c = 0; c < canais; c++) {
						final int offX_lc = offX_l + c * areaX;
						final int offK_fc = offKf + c * areaK;

						for (int kh = 0; kh < altK; kh++) {
							final int off_k_lin = offK_fc + kh * largK;
							final int x_base_h = offX_lc + kh * largX;

							for (int kw = 0; kw < largK; kw++) {
								final double valK = dataK[off_k_lin + kw];
								final int x_base_w = x_base_h + kw;

								for (int i = 0; i < altS; i++) {
									final int out_idx = offY + i * largS;
									final int in_idx = x_base_w + i * largX;

									for (int j = 0; j < largS; j++) {
										dataS[out_idx + j] += dataX[in_idx + j] * valK;
									}
								}
							}
						}
					}
				}
			}));
		}
		
		for (int i = 0; i < tarefas.size(); i++) {
			tarefas.get(i).join();
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

				ops.conv2DFull(
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
	
					ops.corr2D(
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
	
					ops.corr2D(
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
	
		if (JNNNative.jni) {
			JNNNative.conv2dBackward(
				entrada.array(),
				kernel.array(),
				gradS.array(),
				gradK.array(),
				gradB.isPresent() ? gradB.get().array() : null,
				gradB.isPresent(),
				gradE.array(),
				lotes, canais, filtros, 
				altX, largX, 
				altK, largK
			);

			return;
		}

		final int altS = altX - altK + 1;
		final int largS = largX - largK + 1;

		final int areaX = altX * largX;
		final int areaK = altK * largK;
		final int areaGS = altS * largS;

		final double[] dataX = entrada.array();
		final double[] dataK = kernel.array();
		final double[] dataGS = gradS.array();
		final double[] dataGK = gradK.array();
		final double[] dataGE = gradE.array();
		
		final int offXBase = entrada.offset();
		final int offKBase = kernel.offset();
		final int offGSBase = gradS.offset();
		final int offGKBase = gradK.offset();
		final int offGEBase = gradE.offset();

		final boolean temBias = gradB.isPresent();
		final double[] dataGB = temBias ? gradB.get().array() : null;
		final int offGBBase = temBias ? gradB.get().offset() : 0;

		List<ForkJoinTask<?>> tarefas1 = new ArrayList<>(filtros);

		for (int f = 0; f < filtros; f++) {
			final int filtro = f;
			
			tarefas1.add(pool.submit(() -> {
				final int offGKf = offGKBase + filtro * canais * areaK;
				double somaBiasLocal = 0;

				for (int l = 0; l < lotes; l++) {
					final int offGSlf = offGSBase + (l * filtros + filtro) * areaGS;
					final int offXl = offXBase + l * canais * areaX;

					if (temBias) {
						for (int i = 0; i < areaGS; i++) {
							somaBiasLocal += dataGS[offGSlf + i];
						}
					}

					for (int c = 0; c < canais; c++) {
						final int offXc = offXl + c * areaX;
						final int offGKc = offGKf + c * areaK;

						for (int kh = 0; kh < altK; kh++) {
							final int offGKlin = offGKc + kh * largK;
							final int offXh = offXc + kh * largX;

							for (int kw = 0; kw < largK; kw++) {
								final int offXw = offXh + kw;
								double valGrad = 0;
								
								for (int i = 0; i < altS; i++) {
									final int idGS = offGSlf + i * largS;
									final int idX = offXw + i * largX;
									
									for (int j = 0; j < largS; j++) {
										valGrad += dataX[idX + j] * dataGS[idGS + j];
									}
								}
								
								dataGK[offGKlin + kw] += valGrad;
							}
						}
					}
				}
				
				if (temBias) {
					dataGB[offGBBase + filtro] += somaBiasLocal; 
				}
				
			}));
		}

		for (ForkJoinTask<?> t : tarefas1) {
			t.join();
		}

		List<ForkJoinTask<?>> tarefas2 = new ArrayList<>(lotes);

		for (int l = 0; l < lotes; l++) {
			
			final int lote = l;
			tarefas2.add(pool.submit(() -> {
				final int offGEl = offGEBase + lote * canais * areaX;
				final int offGSl = offGSBase + lote * filtros * areaGS;

				for (int c = 0; c < canais; c++) {
					final int offGEc = offGEl + c * areaX;
					
					for (int f = 0; f < filtros; f++) {
						final int offGSf = offGSl + f * areaGS;
						final int offKfc = offKBase + (f * canais + c) * areaK;

						for (int kh = 0; kh < altK; kh++) {
							final int offKlin = offKfc + kh * largK;
							final int offGEh = offGEc + kh * largX;

							for (int kw = 0; kw < largK; kw++) {
								final double valK = dataK[offKlin + kw];
								final int offGEw = offGEh + kw;

								for (int i = 0; i < altS; i++) {
									final int idGE = offGEw + i * largX;
									final int idGS = offGSf + i * largS;

									for (int j = 0; j < largS; j++) {
										dataGE[idGE + j] += dataGS[idGS + j] * valK;
									}
								}
							}
						}
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
		
		if (JNNNative.jni) {
			final int lotes = 1;
			final int altX = entrada.tamDim(1);
			final int largX = entrada.tamDim(2);
			
			JNNNative.maxPool2dForward(
				entrada.array(),
				saida.array(),
				lotes,
				canais, 
				altX, largX,
				filtro[0], filtro[1],
				stride[0], stride[1]
			);
			
			return;
		} 

		for (int i = 0; i < canais; i++) {
			ops.maxPool2D(
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
		final int lotes = entrada.tamDim(0);
		
		if (JNNNative.jni) {
			final int canais = entrada.tamDim(1);
			final int altX = entrada.tamDim(2);
			final int largX = entrada.tamDim(3);

			JNNNative.maxPool2dForward(
				entrada.array(),
				saida.array(),
				lotes,
				canais,
				altX,
				largX,
				filtro[0], filtro[1],
				stride[0], stride[1]
			);

			return;
		}

		// isso aqui vai melhorar ainda
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
		final int canais = entrada.tamDim(0);
		final int altX = entrada.tamDim(1);
		final int largX = entrada.tamDim(2);
		final int altG = grad.tamDim(1);
		final int largG = grad.tamDim(2);

		if (JNNNative.jni) {
			final int lotes = 1;

			JNNNative.maxPool2dBackward(
				entrada.array(),
				grad.array(),
				gradE.array(),
				lotes, 
				canais,
				altX,
				largX,
				altG,
				largG,
				filtro[0], filtro[1],
				stride[0], stride[1]
			);

			return;
		}

		double[] dataE  = entrada.array();
		double[] dataGS = grad.array();
		double[] dataGE = gradE.array();

		int canalSizeEntrada = altX * largX;
		int canalSizeGradS   = altG * largG;
		double val, valMax;

		for (int c = 0; c < canais; c++) {
			int baseEntrada = c * canalSizeEntrada;
			int baseGradS   = c * canalSizeGradS;

			for (int i = 0; i < altG; i++) {
				int linInicio = i * stride[0];
				int linFim    = Math.min(linInicio + filtro[0], altX);

				for (int j = 0; j < largG; j++) {
					int colInicio = j * stride[1];
					int colFim    = Math.min(colInicio + filtro[1], largX);

					valMax = Double.NEGATIVE_INFINITY;
					int linMax = linInicio;
					int colMax = colInicio;

					// Encontrar posição do máximo
					for (int y = linInicio; y < linFim; y++) {
						int idLinha = baseEntrada + y * largX;
						for (int x = colInicio; x < colFim; x++) {
							val = dataE[idLinha + x];
							if (val > valMax) {
								valMax = val;
								linMax = y;
								colMax = x;
							}
						}
					}

					dataGE[baseEntrada + linMax * largX + colMax] += dataGS[baseGradS + i * largG + j];
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
		
		if (JNNNative.jni) {
			final int canais = entrada.tamDim(1);
			final int altX = entrada.tamDim(2);
			final int largX = entrada.tamDim(3);
			final int altG = grad.tamDim(2);
			final int largG = grad.tamDim(3);

			JNNNative.maxPool2dBackward(
				entrada.array(),
				grad.array(),
				gradE.array(),
				lotes,
				canais, 
				altX,
				largX,
				altG,
				largG,
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
			ops.avgPool2D(entrada.subTensor(i), saida.subTensor(i), filtro, stride);
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
