package jnn.core.ops;

import java.util.Optional;

import jnn.core.JNNnative;
import jnn.core.tensor.Tensor;

/**
 * Implementações java em cpu das operações internas da biblioteca.
 */
public final class OpsCPU extends Ops {

    @Override
	public Tensor matmul(Tensor a, Tensor b) {
		return OpsLinear.matmul(a, b);
	}

    @Override
	public void matmul(Tensor a, Tensor b, Tensor dst) {
		if (JNNnative.jni) OpsLinear.matmul_jni(a, b, dst);
		else OpsLinear.matmul(a, b, dst);
	}

    @Override
	public Tensor corr2D(Tensor x, Tensor k) {
		return OpsConv.corr2D(x, k);
	}

    @Override
	public void corr2D(Tensor x, Tensor k, Tensor dst) {
        OpsConv.corr2D(x, k, dst);
	}

	@Override
	public void corr2D(float[] dataX, int offX, float[] dataK, int offK, float[] dataDst, int offDst, int W, int H, int kW, int kH) {
		OpsConv.corr2D(dataX, offX, dataK, offK, dataDst, offDst, W, H, kW, kH);
	}

    @Override
	public Tensor conv2D(Tensor x, Tensor k) {
		return OpsConv.conv2D(x, k);
	}

    @Override
	public void conv2D(Tensor x, Tensor k, Tensor dst) {
		OpsConv.conv2D(x, k, dst);
	}

    @Override
	public Tensor conv2DFull(Tensor x, Tensor k) {
		return OpsConv.conv2DFull(x, k);
	}

    @Override
	public void conv2DFull(Tensor x, Tensor k, Tensor dst) {
		OpsConv.conv2DFull(x, k, dst);
	}

	@Override
	public void conv2DFull(
		float[] dataX, int offX,
		float[] dataK, int offK,
		float[] dataDst, int offDst,
		int W, int H,
		int kW, int kH
	) {
		OpsConv.conv2DFull(dataX, offX, dataK, offK, dataDst, offDst, W, H, kW, kH);
	}

    @Override
	public Tensor maxPool2D(Tensor x, int[] filtro) {
		return OpsPooling.maxPool2D(x, filtro);
	}

    @Override
	public Tensor maxPool2D(Tensor x, int[] filtro, int[] stride) {
		return OpsPooling.maxPool2D(x, filtro, stride);
	}

    @Override
	public void maxPool2D(Tensor x, Tensor dst, int[] filtro, int[] stride) {
		OpsPooling.maxPool2D(x, dst, filtro, stride);
	}

    @Override
	public Tensor avgPool2D(Tensor x, int[] stride) {
		return OpsPooling.avgPool2D(x, stride);
	}

    @Override
	public Tensor avgPool2D(Tensor x, int[] filtro, int[] stride) {
		return OpsPooling.avgPool2D(x, filtro, stride);		
	}

    @Override
	public void avgPool2D(Tensor x, Tensor dst, int[] filtro, int[] stride) {
		OpsPooling.avgPool2D(x, dst, filtro, stride);
	}

	/**
	 * Experimental
	 * @param x {@code Tensor} de entrada.
	 * @param altK altura do kernel (filtro).
	 * @param largK largura do kernel (filtro).
	 * @param altStd altura do stride.
	 * @param largStd largura do stride.
	 * @param altPad altura do padding.
	 * @param largPad largura do padding.
	 * @return {@code Tensor} convertido para o formato {@code im2col}.
	 */
	public Tensor im2col(Tensor x, int altK, int largK, int altStd, int largStd, int altPad, int largPad) {
		int[] shape = x.shape();
		if (shape.length != 3) {
			throw new IllegalArgumentException("O tensor de entrada deve ter formato [C, H, W].");
		}

		int canais = shape[0];
		int altIn = shape[1];
		int largIn = shape[2];

		int altOut = (altIn + 2 * altPad - altK) / altStd + 1;
		int largOut = (largIn + 2 * largPad - largK) / largStd + 1;

		Tensor col = new Tensor(new int[]{canais * altK * largK, altOut * largOut});
		float[] dadosX = x.array();
		float[] dadosC = col.array();

		int canalArea = altK * largK;
		int colunaCol = altOut * largOut;

		// Iteração principal
		for (int c = 0; c < canais; c++) {
			for (int kh = 0; kh < altK; kh++) {
				for (int kw = 0; kw < largK; kw++) {
					int linhaBase = c * canalArea + kh * largK + kw;
					int outIndex = 0;
					int inY0 = kh - altPad;

					for (int y = 0; y < altOut; y++) {
						int inY = inY0 + y * altStd;
						int inX0 = kw - largPad;

						for (int x2 = 0; x2 < largOut; x2++) {
							int inX = inX0 + x2 * largStd;

							if (inY >= 0 && inY < altIn && inX >= 0 && inX < largIn) {
								dadosC[linhaBase * colunaCol + outIndex] = dadosX[c * altIn * largIn + inY * largIn + inX];
							} else {
								dadosC[linhaBase * colunaCol + outIndex] = 0.0f;
							}

							outIndex++;
						}
					}
				}
			}
		}

		return col;
	}

	/**
	 * Experimental
	 * @param entrada
	 * @param kernel
	 * @param bias
	 * @param saida
	 */
	public void forwardConv2DIm2col(Tensor entrada, Tensor kernel, Optional<Tensor> bias, Tensor saida) {
		int[] kShape = kernel.shape();// (filtros, canais, kH, kW)
		int numFiltros = kShape[0];
		int canais = kShape[1];
		int kH = kShape[2];
		int kW = kShape[3];
		int padH = 0; 
		int padW = 0;
		int strideH = 1;
		int strideW = 1;

		int H = entrada.tamDim(1);
		int W = entrada.tamDim(2);
		int outH = (H + 2 * padH - kH) / strideH + 1;
		int outW = (W + 2 * padW - kW) / strideW + 1;

		Tensor im2Col = im2col(entrada, kH, kW, strideH, strideW, padH, padW);
		Tensor flatK = kernel.reshape(numFiltros, canais * kH * kW); 

		Tensor res = new Tensor(numFiltros, outH * outW);
		matmul(flatK, im2Col, res);

		res = res.reshape(numFiltros, outH, outW);
		saida.copiar(res);
		
		bias.ifPresent(b -> {
			for (int f = 0; f < numFiltros; f++) {
				float x = b.get(f);
				saida.subTensor(f).add(x);
			}
		});
	}

}
