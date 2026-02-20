package jnn.core.ops;

import jnn.core.tensor.Tensor;

/**
 * Implementações internas de operações de pooling.
 */
public class OpsPooling {

	/**
	 * Construtor privado.
	 */
	private OpsPooling() {}
    
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

	/**
	 * Operação interna de max pooling 2D.
	 * @param x {@code Tensor} de entrada.
	 * @param filtro formato do filtro (altura, largura).
	 * @return {@code Tensor} resultado.
	 */
	public static Tensor maxPool2D(Tensor x, int[] filtro) {
		return maxPool2D(x, filtro, filtro);// stride = filtro
	}

	/**
	 * Operação interna de max pooling 2D.
	 * @param x {@code Tensor} de entrada.
	 * @param filtro formato do filtro (altura, largura).
	 * @param stride formato dos strides (altura, largura).
	 * @return {@code Tensor} resultado.
	 */
	public static Tensor maxPool2D(Tensor x, int[] filtro, int[] stride) {
		if (x.numDim() != 2) {
			throw new IllegalArgumentException(
				"\nEntrada deve ser 2D, mas é " + x.numDim() + "D."
			);
		}

		if (filtro.length != 2) {
			throw new IllegalArgumentException(
				"\nFormato do filtro deve conter dois elementos, " +
				" recebido " + filtro.length
			);
		}

		if (stride.length != 2) {
			throw new IllegalArgumentException(
				"\nFormato do stride deve conter dois elementos, " +
				" recebido " + filtro.length
			);
		}

		int[] shapeEntrada = x.shape();

		int[] poolShape = calcShapeConv(
			shapeEntrada, 
			filtro, 
			stride
		);

		Tensor pool = new Tensor(poolShape[0], poolShape[1]);
		maxPool2D(x, pool, filtro, stride);

		return pool;
	}

	/**
	 * Operação interna de max pooling 2D.
	 * @param x {@code Tensor} de entrada.
	 * @param dst {@code Tensor} de destino.
	 * @param filtro formato do filtro (altura, largura).
	 * @param stride formato dos strides (altura, largura).
	 */
	public static void maxPool2D(Tensor x, Tensor dst, int[] filtro, int[] stride) {
		if (x.numDim() != 2 || dst.numDim() != 2) {
			throw new IllegalArgumentException(
				"maxPool2D agora aceita apenas tensores 2D. Entrada=" +
				x.numDim() + "D, Saída=" + dst.numDim() + "D."
			);
		}

		if (filtro.length != 2) {
			throw new IllegalArgumentException("Filtro deve ser [fH, fW]");
		}

		if (stride.length != 2) {
			throw new IllegalArgumentException("Stride deve ser [sH, sW]");
		}

		int H = x.shape()[0];
		int W = x.shape()[1];

		int H_out = dst.shape()[0];
		int W_out = dst.shape()[1];

		int[] esperado = calcShapeConv(new int[]{H, W}, filtro, stride);
		if (esperado[0] != H_out || esperado[1] != W_out) {
			throw new IllegalArgumentException(
				"Saída esperada = [" + esperado[0] + "," + esperado[1] +
				"] mas recebeu " + dst.shapeStr()
			);
		}

		float[] arrIn  = x.array();
		float[] arrOut = dst.array();

		int baseOffsetIn  = x.offset();
		int baseOffsetOut = dst.offset();

		int inStrideH  = x.strides()[0];
		int inStrideW  = x.strides()[1];
		int outStrideH = dst.strides()[0];
		int outStrideW = dst.strides()[1];

		int fH = filtro[0];
		int fW = filtro[1];
		int sH = stride[0];
		int sW = stride[1];

		float maxVal, val;

		for (int i = 0; i < H_out; i++) {
			int linInicio = i * sH;
			int linFim = Math.min(linInicio + fH, H);

			for (int j = 0; j < W_out; j++) {
				int colInicio = j * sW;
				int colFim = Math.min(colInicio + fW, W);
				maxVal = Float.NEGATIVE_INFINITY;

				for (int l = linInicio; l < linFim; l++) {
					int baseLinha = baseOffsetIn + l * inStrideH;
					for (int m = colInicio; m < colFim; m++) {
						val = arrIn[baseLinha + m * inStrideW];
						if (val > maxVal) maxVal = val;
					}
				}

				arrOut[baseOffsetOut + i * outStrideH + j * outStrideW] = maxVal;
			}
		}
	}

	/**
	 * Operação interna de avg pooling 2D.
	 * @param x {@code Tensor} de entrada.
	 * @param filtro formato do filtro (altura, largura).
	 * @return {@code Tensor} resultado.
	 */
	public static Tensor avgPool2D(Tensor x, int[] filtro) {
		return avgPool2D(x, filtro, filtro);// stride = filtro
	}

	/**
	 * Operação interna de avg pooling 2D.
	 * @param x {@code Tensor} de entrada.
	 * @param filtro formato do filtro (altura, largura).
	 * @param stride formato dos strides (altura, largura).
	 * @return {@code Tensor} resultado.
	 */
	public static Tensor avgPool2D(Tensor x, int[] filtro, int[] stride) {
		if (x.numDim() != 2) {
			throw new IllegalArgumentException(
				"\nEntrada deve ser 2D, mas é " + x.numDim() + "D."
			);
		}

		if (filtro.length != 2) {
			throw new IllegalArgumentException(
				"\nFormato do filtro deve conter dois elementos, " +
				" recebido " + filtro.length
			);
		}

		if (stride.length != 2) {
			throw new IllegalArgumentException(
				"\nFormato do stride deve conter dois elementos, " +
				" recebido " + filtro.length
			);
		}

		int[] shapeEntrada = x.shape();

		int[] poolShape = calcShapeConv(
			shapeEntrada, 
			filtro, 
			stride
		);

		Tensor pool = new Tensor(poolShape[0], poolShape[1]);
		avgPool2D(x, pool, filtro, stride);

		return pool;		
	}

	/**
	 * Operação interna de avg pooling 2D.
	 * @param x {@code Tensor} de entrada.
	 * @param dst {@code Tensor} de destino.
	 * @param filtro formato do filtro (altura, largura).
	 * @param stride formato dos strides (altura, largura).
	 */
	public static void avgPool2D(Tensor x, Tensor dst, int[] filtro, int[] stride) {
		if (x.numDim() != 2 || dst.numDim() != 2) {
			throw new UnsupportedOperationException(
				"\nAmbos os tensores devem ser 2D, recebido " +
				" entrada = " + x.numDim() + "D e saida = " + dst.numDim() + "D."
			);
		}

		if (filtro.length != 2) {
			throw new IllegalArgumentException(
				"\nFormato do filtro deve conter dois elementos, " +
				" recebido " + filtro.length
			);
		}

		if (stride.length != 2) {
			throw new IllegalArgumentException(
				"\nFormato do stride deve conter dois elementos, " +
				" recebido " + filtro.length
			);
		}

		int[] shapeEntrada = x.shape();
		int[] shapeSaida   = dst.shape();

		int altEntrada  = shapeEntrada[0];
		int largEntrada = shapeEntrada[1];
		int altSaida    = shapeSaida[0];
		int largSaida   = shapeSaida[1];

		int[] shapeEsp = calcShapeConv(
			shapeEntrada, 
			filtro,
			stride
		);
		
		if (altSaida != shapeEsp[0] || largSaida != shapeEsp[1]) {
			throw new IllegalArgumentException(
				"\nDimensão de saída esperada (" + shapeEsp[0] + ", " + shapeEsp[1] + "), mas" +
				" recebido " + dst.shapeStr()
			);
		}

		float[] dataE = x.array();
		float[] dataS = dst.array();

		int offE = x.offset();
		int[] strE = x.strides();

		int offS = dst.offset();
		int[] strS = dst.strides();

		int fH = filtro[0];
		int fW = filtro[1];
		int sH = stride[0];
		int sW = stride[1];

		for (int i = 0; i < altSaida; i++) {
			int linInicio = i * sH;
			int linFim = Math.min(linInicio + fH, altEntrada);

			for (int j = 0; j < largSaida; j++) {
				int colInicio = j * sW;
				int colFim = Math.min(colInicio + fW, largEntrada);

				float soma = 0.0f;
				int cont = 0;
				for (int l = linInicio; l < linFim; l++) {
					int baseLinha = offE + l * strE[0];
					for (int m = colInicio; m < colFim; m++) {
						soma += dataE[baseLinha + m * strE[1]];
						cont++;
					}
				}
				
				dataS[offS + i * strS[0] + j * strS[1]] = soma / cont;
			}
		}
		
	}

}
