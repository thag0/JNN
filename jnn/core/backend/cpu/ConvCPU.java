package jnn.core.backend.cpu;

import jnn.core.tensor.Tensor;

public class ConvCPU {
    
	private static int[] calcShapeConv(int[] entrada, int[] filtro, int[] stride) {
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

	public static Tensor corr2D(Tensor x, Tensor k) {
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

	public static void corr2D(Tensor x, Tensor k, Tensor dst) {
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

		double[] dataE = x.array();
		double[] dataK = k.array();
		double[] dataS = dst.array();

		// lidar com views de tensores
		int offE = x.offset();
		int offK = k.offset();
		int offS = dst.offset();

		double soma;
		for (int i = 0; i < altEsp; i++) {
			for (int j = 0; j < largEsp; j++) {

				soma = 0.0;
				final int idSaida = offS + (i * largEsp + j);
				for (int l = 0; l < altKernel; l++) {
					int idBaseEntrada = offE + ((l + i) * largEntrada);
					int idBaseKernel  = offK + (l * largKernel); 
					for (int m = 0; m < largKernel; m++) {
						soma += 
							dataE[idBaseEntrada + (m + j)] *
							dataK[idBaseKernel + m];
					}
				}

				dataS[idSaida] += soma;
			}
		}

	}

	public static Tensor conv2D(Tensor x, Tensor k) {
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

	public static void conv2D(Tensor x, Tensor k, Tensor dst) {
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

		double[] dataE = x.array();
		double[] dataK = k.array();
		double[] dataS = dst.array();

		int offE = x.offset();
		int offK = k.offset();
		int offS = dst.offset();
	
		double soma;
		for (int i = 0; i < altEsp; i++) {
			for (int j = 0; j < largEsp; j++) {
				
				soma = 0.0;
				final int idSaida = offS + (i * largEsp + j);
				for (int l = 0; l < altKernel; l++) {
					for (int m = 0; m < largKernel; m++) {
						soma +=
							dataE[offE + ((l + i) * largEntrada + (m + j))] * 
							dataK[offK + ((altKernel - 1 - l) * largKernel + (largKernel - 1 - m))];
					}
				}

				dataS[idSaida] += soma;
			}
		}
		
	}

	public static Tensor conv2DFull(Tensor x, Tensor k) {
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

	public static void conv2DFull(Tensor entrada, Tensor kernel, Tensor saida) {
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

		double[] dataE = entrada.array();
		double[] dataK = kernel.array();
		double[] dataS = saida.array();

		// lidar com views de tensores
		int offE = entrada.offset();
		int offK = kernel.offset();
		int offS = saida.offset();

		double soma;
		for (int i = 0; i < altEsp; i++) {
			for (int j = 0; j < largEsp; j++) {
				
				soma = 0.0;
				final int idSaida = offS + (i * largEsp + j);
				for (int m = 0; m < altKernel; m++) {
					int linEntrada = i - m;
					if (linEntrada >= 0 && linEntrada < altEntrada) {
						for (int n = 0; n < largKernel; n++) {
							int colEntrada = j - n;
							if (colEntrada >= 0 && colEntrada < largEntrada) {
								soma +=
									dataK[offK + (m * largKernel + n)] *
									dataE[offE + (linEntrada * largEntrada + colEntrada)];
							}
						}
					}
				}

				dataS[idSaida] += soma;
			}
		}

	}

}
