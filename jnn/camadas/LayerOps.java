package jnn.camadas;

import java.util.List;
import java.util.ArrayList;
import java.util.Optional;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;

import jnn.core.JNNnative;
import jnn.core.ops.Ops;
import jnn.core.parallel.PoolFactory;
import jnn.core.tensor.Tensor;

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
	private final ForkJoinPool pool = PoolFactory.common();

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
	 * @see jnn.camadas.Densa Densa
	 */
	public void forwardDensa(Tensor entrada, Tensor kernel, Optional<Tensor> bias, Tensor saida) {
		saida.zero();// zerar acumulos anteriores

		ops.matmul(entrada, kernel, saida);

		bias.ifPresent(b -> {
			if (entrada.numDim() == 1) {//amostra única
				saida.add(b);

			} else if (entrada.numDim() == 2) {//lote de amostras
				saida.copiar(
					saida.broadcast(b, (_s, _b) -> _s + _b)
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
	 * @see jnn.camadas.Densa Densa
	 */
	public void backwardDensa(Tensor entrada, Tensor kernel, Tensor gradS, Tensor gradK, Optional<Tensor> gradB, Tensor gradE) {
		gradE.zero();// zerar acumulaçoes anteriores
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
	 * @param padding {@code array} contendo o formato de padding (altura, largura)
	 * @see jnn.camadas.Conv2D Conv2D
	 */
	public void forwardConv2D(
		Tensor entrada,
		Tensor kernel,
		Optional<Tensor> bias,
		Tensor saida,
		int[] padding) {

		int dims = entrada.numDim();
		if (dims != 3 && dims != 4) {
			throw new IllegalArgumentException(
				"\nEntrada esperada 3D (C, H, W) ou 4D (N, C, H, W), recebido " + dims + "D."
			);
		}

		if (dims == 3) {
			entrada = entrada.unsqueeze(0);
			saida = saida.unsqueeze(0);
		}
		
		runConv2DForward(entrada, kernel, bias, saida, padding);
		
		if (dims == 3) {
			entrada = entrada.squeeze(0);
			saida = saida.squeeze(0);
		}
	}

	/**
	 * Realiza a operação de forward da camada Conv2D usando o formato NCHW.
	 * @param entrada entrada.
	 * @param kernel kernel.
	 * @param bias bias.
	 * @param saida saída.
	 * @param padding padding.
	 */
	private void runConv2DForward(
		Tensor entrada,
		Tensor kernel,
		Optional<Tensor> bias,
		Tensor saida,
		int[] padding) {
		
		final int[] shapeX = entrada.shape();
		final int[] shapeK = kernel.shape();
		final int lotes = shapeX[0];
		final int canais = shapeX[1];
		final int filtros = shapeK[0];
		final int altX = shapeX[2];
		final int largX = shapeX[3];
		
		final int altK = shapeK[2];
		final int largK = shapeK[3];

		final int altPad = padding[0];
		final int largPad = padding[1];

		if (JNNnative.isOn()) {
			JNNnative.conv2dForward(
				entrada.array(),
				kernel.array(),
				bias.isPresent() ? bias.get().array() : null,
				bias.isPresent(),
				saida.array(),
				lotes, canais, filtros,
				altX, largX,
				altK, largK,
				altPad, largPad
			);

			return;
		}

		final int altS  = altX - altK + 1 + 2 * altPad;
		final int largS = largX - largK + 1 + 2 * largPad;

		final int areaX = altX * largX;
		final int areaK = altK * largK;
		final int areaS = altS * largS;

		final float[] dataX = entrada.array();
		final float[] dataK = kernel.array();
		final float[] dataS = saida.array();
		final float[] dataB = bias.isPresent() ? bias.get().array() : null;

		final int offXBase = entrada.offset();
		final int offKBase = kernel.offset();
		final int offYBase = saida.offset();

		var tasks = new ArrayList<ForkJoinTask<?>>(filtros);

		for (int f = 0; f < filtros; f++) {
			final int filtro = f;
			final int offKf = offKBase + filtro * canais * areaK;
			final float valBias = (dataB != null) ? dataB[filtro] : 0.0f;
			
			tasks.add(pool.submit(() -> {
				for (int l = 0; l < lotes; l++) {
					final int offY = offYBase + (l * filtros + filtro) * areaS;
					final int offXL = offXBase + (l * canais) * areaX;

					for (int i = 0; i < areaS; i++) {
						dataS[offY + i] = valBias;
					}

					for (int c = 0; c < canais; c++) {
						final int offXLc = offXL + c * areaX;
						final int offKFc = offKf + c * areaK;

						for (int kh = 0; kh < altK; kh++) {
							final int i_min = Math.max(altPad - kh, 0);
							final int i_max = Math.min(altX + altPad - kh, altS);

							for (int kw = 0; kw < largK; kw++) {
								final int j_min = Math.max(largPad - kw, 0);
								final int j_max = Math.min(largX + largPad - kw, largS);
								final float valK = dataK[offKFc + kh * largK + kw];								

								for (int i = i_min; i < i_max; i++) {
									final int in_y = i + kh - altPad;
									final int baseDst = offY + i * largS;
									final int baseX   = offXLc + in_y * largX;

									for (int j = j_min; j < j_max; j++) {
										final int in_x = j + kw - largPad;
										dataS[baseDst + j] += dataX[baseX + in_x] * valK;
									}
								}
							}
						}
					}
				}
			}));
		}
		
		for (var task : tasks) task.join();
	}

	/**
	 * Realiza a propagação reversa através da camada convolucional.
	 * @param entrada {@code Tensor} contendo a entrada da camada.
	 * @param kernel {@code Tensor} contendos o kernel/filtros da camada.
	 * @param gradS {@code Tensor} contendo o gradiente em relação a saída da camada.
	 * @param gradK {@code Tensor} contendo o gradiente em relação ao kernel/filtros da camada.
	 * @param gradB {@code Tensor} contendo o gradiente em relação ao bias da camada {@code (podendo ser nulo)}.
	 * @param gradE {@code Tensor} contendo o gradiente em relação à entrada da camada.
	 * @param padding {@code array} contendo o formato de padding (altura, largura)
	 * @see jnn.camadas.Conv2D Conv2D
	 */
	public void backwardConv2D(
		Tensor entrada, 
		Tensor kernel, 
		Tensor gradS, 
		Tensor gradK, 
		Optional<Tensor> gradB, 
		Tensor gradE, 
		int[] padding) {

		int dims = entrada.numDim();
		if (dims != 3 && dims != 4) {
			throw new IllegalArgumentException(
				"\nEntrada esperada 3D (C, H, W) ou 4D (N, C, H, W), recebido " + dims + "D."
			);
		}

		if (dims == 3) {
			entrada = entrada.unsqueeze(0);
			gradE = gradE.unsqueeze(0);
			gradS = gradS.unsqueeze(0);
		}
		
		runConv2DBackward(entrada, kernel, gradS, gradK, gradB, gradE, padding);
		
		if (dims == 3) {
			entrada = entrada.squeeze(0);
			gradE = gradE.squeeze(0);
			gradS = gradS.squeeze(0);
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
	 * @param padding padding
	 */
	private void runConv2DBackward(
		Tensor entrada,
		Tensor kernel,
		Tensor gradS,
		Tensor gradK,
		Optional<Tensor> gradB,
		Tensor gradE,
		int[] padding) {

		final int[] shapeX = entrada.shape();
		final int[] shapeK = kernel.shape();

		final int lotes = shapeX[0];
		final int canais = shapeX[1];
		final int altX = shapeX[2];
		final int largX = shapeX[3];
	
		final int filtros  = shapeK[0];
		final int altK = shapeK[2];
		final int largK = shapeK[3];

		final int altPad = padding[0];
		final int largPad = padding[1];
		
		gradE.zero();// zerar acumulaçoes anteriores

		if (JNNnative.isOn()) {
			JNNnative.conv2dBackward(
				entrada.array(),
				kernel.array(),
				gradS.array(),
				gradK.array(),
				gradB.isPresent() ? gradB.get().array() : null,
				gradB.isPresent(),
				gradE.array(),
				lotes, canais, filtros, 
				altX, largX, 
				altK, largK,
				altPad, largPad
			);

			return;
		}

		final int[] shapeGS = gradS.shape();
		final int altS = shapeGS[2]; 
    	final int largS = shapeGS[3];

		final int areaX = altX * largX;
		final int areaK = altK * largK;
		final int areaGS = altS * largS;

		final float[] dataX = entrada.array();
		final float[] dataK = kernel.array();
		final float[] dataGS = gradS.array();
		final float[] dataGK = gradK.array();
		final float[] dataGE = gradE.array();
		
		final int offXBase = entrada.offset();
		final int offKBase = kernel.offset();
		final int offGSBase = gradS.offset();
		final int offGKBase = gradK.offset();
		final int offGEBase = gradE.offset();

		final boolean temBias = gradB.isPresent();
		final float[] dataGB = temBias ? gradB.get().array() : null;
		final int offGBBase = temBias ? gradB.get().offset() : 0;

		List<ForkJoinTask<?>> tarefas1 = new ArrayList<>(filtros);

		for (int f = 0; f < filtros; f++) {
			final int filtro = f;
			tarefas1.add(pool.submit(() -> {
		
				if (temBias) {
					float soma = 0f;
					for (int l = 0; l < lotes; l++) {
						final int offGSlf = offGSBase + (l * filtros + filtro) * areaGS;
						for (int i = 0; i < areaGS; i++) {
							soma += dataGS[offGSlf + i];
						}
					}
					dataGB[offGBBase + filtro] += soma;
				}

				for (int c = 0; c < canais; c++) {
					final int offGK_fc = offGKBase + (filtro * canais + c) * areaK;

					for (int kh = 0; kh < altK; kh++) {
						for (int kw = 0; kw < largK; kw++) {
							int iMin = Math.max(0, altPad - kh);
							int iMax = Math.min(altS, altX + altPad - kh);
							int jMin = Math.max(0, largPad - kw);
							int jMax = Math.min(largS, largX + largPad - kw);
							float soma = 0f;

							for (int l = 0; l < lotes; l++) {
								final int offGS_loteFilter = offGSBase + (l * filtros + filtro) * areaGS;
								final int offX_loteCanal   = offXBase + (l * canais + c) * areaX;

								for (int hOut = iMin; hOut < iMax; hOut++) {
									int inH = hOut + kh - altPad;
									
									final int idxGS_Line = offGS_loteFilter + hOut * largS;
									final int idxX_Line  = offX_loteCanal   + inH * largX;
									
									int inW_Offset = kw - largPad;

									for (int wOut = jMin; wOut < jMax; wOut++) {
										float valGS = dataGS[idxGS_Line + wOut];
										float valX  = dataX[idxX_Line + (wOut + inW_Offset)];
										soma += valX * valGS;
									}
								}
							}
							dataGK[offGK_fc + kh * largK + kw] += soma;
						}
					}
				}
			}));
		}
		
		for (ForkJoinTask<?> t : tarefas1) t.join();

		List<ForkJoinTask<?>> tarefas2 = new ArrayList<>(lotes);

		for (int l = 0; l < lotes; l++) {
			final int lote = l;

			tarefas2.add(pool.submit(() -> {
				final int offGE_lote = offGEBase + lote * canais * areaX;
				final int offGS_lote = offGSBase + lote * filtros * areaGS;

				for (int c = 0; c < canais; c++) {
					final int offGE_loteCanal = offGE_lote + c * areaX;

					for (int f = 0; f < filtros; f++) {
						final int offGS_loteFilter = offGS_lote + f * areaGS;
						final int offK_filterCanal = offKBase + (f * canais + c) * areaK;

						for (int kh = 0; kh < altK; kh++) {
							for (int kw = 0; kw < largK; kw++) {
								float valK = dataK[offK_filterCanal + kh * largK + kw];
								int iStart = Math.max(0, altPad - kh);
								int iEnd   = Math.min(altS, altX + altPad - kh);
								int jStart = Math.max(0, largPad - kw);
								int jEnd   = Math.min(largS, largX + largPad - kw);
								int inW_Offset = kw - largPad;

								for (int outH = iStart; outH < iEnd; outH++) {
									int inH = outH + kh - altPad;
									int idxGE_Line = offGE_loteCanal + inH * largX;
									int idxGS_Line = offGS_loteFilter + outH * largS;

									for (int outW = jStart; outW < jEnd; outW++) {
										dataGE[idxGE_Line + (outW + inW_Offset)] += dataGS[idxGS_Line + outW] * valK;
									}
								}
							}
						}
					}
				}
			}));
		}

		for (var t : tarefas2) t.join();

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
		
		if (JNNnative.isOn()) {
			final int lotes = 1;
			final int altX = entrada.tamDim(1);
			final int largX = entrada.tamDim(2);
			
			JNNnative.maxPool2dForward(
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
		
		if (JNNnative.isOn()) {
			final int canais = entrada.tamDim(1);
			final int altX = entrada.tamDim(2);
			final int largX = entrada.tamDim(3);

			JNNnative.maxPool2dForward(
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

		if (JNNnative.isOn()) {
			final int lotes = 1;

			JNNnative.maxPool2dBackward(
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

		float[] dataE  = entrada.array();
		float[] dataGS = grad.array();
		float[] dataGE = gradE.array();

		int canalSizeEntrada = altX * largX;
		int canalSizeGradS   = altG * largG;
		float val, valMax;

		for (int c = 0; c < canais; c++) {
			int baseEntrada = c * canalSizeEntrada;
			int baseGradS   = c * canalSizeGradS;

			for (int i = 0; i < altG; i++) {
				int linInicio = i * stride[0];
				int linFim    = Math.min(linInicio + filtro[0], altX);

				for (int j = 0; j < largG; j++) {
					int colInicio = j * stride[1];
					int colFim    = Math.min(colInicio + filtro[1], largX);

					valMax = Float.NEGATIVE_INFINITY;
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
		
		if (JNNnative.isOn()) {
			final int canais = entrada.tamDim(1);
			final int altX = entrada.tamDim(2);
			final int largX = entrada.tamDim(3);
			final int altG = grad.tamDim(2);
			final int largG = grad.tamDim(3);

			JNNnative.maxPool2dBackward(
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

		float[] arrGo  = grad.array();
		float[] arrGi  = gradE.array();

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
					float g = arrGo[baseGo + i * stGo[1] + j * stGo[2]] / janela;

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

	/**
	 * Realiza a propagação direta através da camada GlobalAvgPool2D.
	 * @param entrada {@code Tensor} entrada da camada.
	 * @param saida {@code Tensor} saída da camada.
	 * @param shapeIn formato base de entrada da camada.
	 */
	public void forwardGAP(Tensor entrada, Tensor saida, int[] shapeIn) {
        final float[] x  = entrada.array();
        final float[] y = saida.array();

        final int canais = shapeIn[0];
        final int altura = shapeIn[1];
        final int largura = shapeIn[2];
        final int area = altura * largura;
        final int lotes = (entrada.numDim() == 4) ? entrada.tamDim(0) : 1;

        int idX = 0;
        int idY = 0;
        for (int n = 0; n < lotes; n++) {
            for (int c = 0; c < canais; c++) {
                float soma = 0.f;

                for (int i = 0; i < area; i++) {
                    soma += x[idX++];
                }

                y[idY++] = soma / area;
            }
        }
	}

	/**
	 * Realiza a propagação reversa através da camada GlobalAvgPool2D.
	 * @param gradE {@code Tensor} contendo o gradiente em relação a entrada da camada.
	 * @param grad {@code Tensor} contendo o gradiente da saída da camada.
	 * @param shapeIn formato base de entrada da camada.
	 * @param tamLote tamanho de lotes usado na camada.
	 */
	public void backwardGAP(Tensor gradE, Tensor grad, int[] shapeIn, int tamLote) {
        final float[] ge  = gradE.array();
        final float[] gs = grad.array();

        final int canais = shapeIn[0];
        final int altura = shapeIn[1];
        final int largura = shapeIn[2];
        final int area = altura * largura;
        final int lotes = (grad.numDim() == 2) ? tamLote : 1;

        int idGS = 0;
        int idGE  = 0;
        for (int n = 0; n < lotes; n++) {
            for (int c = 0; c < canais; c++) {
                float g = gs[idGS++] / area;

                for (int i = 0; i < area; i++) {
                    ge[idGE++] = g;
                }
            }
        }
	}

	/**
	 * Realiza a propagação direta através da camada BatchNorm2D.
	 * @param entrada entrada da camada.
	 * @param gamma escala.
	 * @param beta deslocamento.
	 * @param entradaNorm entrada normalizada.
	 * @param mediaMovel media movel por canal.
	 * @param varianciaMovel variancia movel por canal.
	 * @param media média por canal.
	 * @param variancia variancia por canal.
	 * @param saida saída da camada.
	 * @param momentum momentum.
	 * @param eps epsilon.
	 * @param treinando controlador de treino/teste.
	 */
	public void forwardBatchNorm2D(
		Tensor entrada,
		Tensor gamma,
		Tensor beta,
		Tensor entradaNorm,
		Tensor mediaMovel,
		Tensor varianciaMovel,
		Tensor media,
		Tensor variancia,
		Tensor saida,
		float momentum,
		float eps,
		boolean treinando) {
			
		boolean entrada3D = entrada.numDim() == 3;

        int lotes	= entrada3D ? 					  1  : entrada.tamDim(0);
        int canais	= entrada3D ? entrada.tamDim(0) : entrada.tamDim(1);
        int altX	= entrada3D ? entrada.tamDim(1) : entrada.tamDim(2);
        int largX	= entrada3D ? entrada.tamDim(2) : entrada.tamDim(3);

        float[] dataX = entrada.array();
        float[] dataY = saida.array();
        float[] dataGamma = gamma.array();
        float[] dataBeta  = beta.array();
        float[] dataRM = mediaMovel.array();
        float[] dataRV = varianciaMovel.array();

		float[] dataMedia = media.array();
		float[] dataVar  = variancia.array();
		float[] dataXNorm = entradaNorm.array();

		if (JNNnative.isOn()) {
			JNNnative.batchNorm2DForward(
				dataX,
				dataY,
				dataGamma,
				dataBeta,
				dataRM,
				dataRV,
				dataMedia,
				dataVar,
				dataXNorm,
				lotes, canais,
				altX, largX,
				momentum,
				eps,
				treinando
			);

			return;
		}

        int areaX = altX * largX;
        int M = lotes * areaX;

        if (treinando) {
            for (int c = 0; c < canais; c++) {
                double soma = 0;
                for (int n = 0; n < lotes; n++) {
                    int base = n * canais * areaX + c * areaX;
                    for (int i = 0; i < areaX; i++) {
                        soma += dataX[base + i];
                    }
                }
                dataMedia[c] = (float) soma / M;
            }

            for (int c = 0; c < canais; c++) {
                float valMed = dataMedia[c];
                double soma = 0;
                
                for (int n = 0; n < lotes; n++) {
                    int base = n * canais * areaX + c * areaX;
                    for (int i = 0; i < areaX; i++) {
                        float dif = dataX[base + i] - valMed;
                        soma += dif * dif;
                    }
                }
                dataVar[c] = (float) soma / M;
            }

            for (int c = 0; c < canais; c++) {
                float invStd = 1.0f / (float)Math.sqrt(dataVar[c] + eps);
                float g = dataGamma[c];
                float b = dataBeta[c];
                float m = dataMedia[c];

                for (int n = 0; n < lotes; n++) {
                    int base = n * canais * areaX + c * areaX;
                    for (int i = 0; i < areaX; i++) {
                        int id = base + i;
                        float norm = (dataX[id] - m) * invStd;
                        dataXNorm[id] = norm;
                        dataY[id]  = g * norm + b;
                    }
                }

                float varDesviezada = (M > 1) ? (dataVar[c] * M / (M - 1f)) : dataVar[c];
                dataRM[c] = (1 - momentum) * dataRM[c] + momentum * dataMedia[c];
                dataRV[c] = (1 - momentum) * dataRV[c]  + momentum * varDesviezada;
            }

        } else {
            for (int c = 0; c < canais; c++) {
                float invStd = 1.0f / (float)Math.sqrt(dataRV[c] + eps);
                float g = dataGamma[c];
                float b = dataBeta[c];
                float m = dataRM[c];

                for (int n = 0; n < lotes; n++) {
                    int base = n*canais*areaX + c*areaX;

                    for (int i = 0; i < areaX; i++) {
                        int id = base + i;
                        dataY[id] = g * (dataX[id] - m) * invStd + b;
                    }
                }
            }
        }

	}

	/**
	 * Realiza a propagação reversa através da camada BatchNorm2D.
	 * @param gradE gradiente em relação a entrada da camada.
	 * @param entradaNorm entrada normalizada.
	 * @param variancia variancia por canal.
	 * @param gamma escala.
	 * @param gradGamma gradiente da escala.
	 * @param gradBeta gradiente do deslocamento.
	 * @param gradS gradiente em relação a saída da camada.
	 * @param eps epsilon.
	 */
	public void backwardBatchNorm2D(
		Tensor gradE,
		Tensor entradaNorm,
		Tensor variancia,
		Tensor gamma,
		Tensor gradGamma,
		Tensor gradBeta,
		Tensor gradS,
		float eps) {

        gradE.zero();

		boolean entrada3D = gradE.numDim() == 3;

        int lotes	= entrada3D ?					1  : gradE.tamDim(0);
        int canais	= entrada3D ? gradE.tamDim(0) : gradE.tamDim(1);
        int altura	= entrada3D ? gradE.tamDim(1) : gradE.tamDim(2);
        int largura	= entrada3D ? gradE.tamDim(2) : gradE.tamDim(3);
		
        float[] dataXNorm = entradaNorm.array();
        float[] dataVar = variancia.array();
        float[] dataGamma = gamma.array();
		
        float[] dataGE = gradE.array();
        float[] dataGS = gradS.array();
        float[] dataGG = gradGamma.array();
        float[] dataGB = gradBeta.array();
		
		if (JNNnative.isOn()) {
			JNNnative.batchNorm2DBackward(
				dataXNorm,
				dataVar,
				dataGamma,
				dataGE,
				dataGS,
				dataGG,
				dataGB,
				lotes,
				canais,
				altura,
				largura, 
				eps
			);

			return;
		}

		int areaX = altura * largura;
		int M = lotes * areaX;
		
        for (int c = 0; c < canais; c++) {
            double somaGamma = 0;
            double somaBeta  = 0;

            for (int n = 0; n < lotes; n++) {
                int base = n * canais * areaX + c * areaX;

                for (int i = 0; i < areaX; i++) {
                    int id = base + i;
                    float grad = dataGS[id];
                    somaGamma += grad * dataXNorm[id];
                    somaBeta  += grad;
                }
            }

            dataGG[c] += (float) somaGamma;
            dataGB[c] += (float) somaBeta;
        }

        for (int c = 0; c < canais; c++) {
            float valGamma = dataGamma[c];
            float invStd = 1.0f / (float)Math.sqrt(dataVar[c] + eps);

            double somaG = 0;
            double somaGENorm = 0;

            for (int n = 0; n < lotes; n++) {
                int base = n * canais * areaX + c * areaX;

                for (int i = 0; i < areaX; i++) {
                    int id = base + i;
                    float grad = dataGS[id];
                    somaG += grad;
                    somaGENorm += grad * dataXNorm[id];
                }
            }

            float coef = valGamma * invStd / M;

            for (int n = 0; n < lotes; n++) {
                int base = n * canais * areaX + c * areaX;

                for (int i = 0; i < areaX; i++) {
                    int idx = base + i;
                    float grad = dataGS[idx];
                    float xNorm = dataXNorm[idx];

                    dataGE[idx] = (float) (coef * (M * grad - somaG - xNorm * somaGENorm));
                }
            }
        }

	}

}
