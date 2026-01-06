package jnn.core.backend.cpu;

import jnn.core.backend.Backend;
import jnn.core.tensor.Tensor;

/**
 * Implementações em cpu das operações internas da biblioteca.
 */
public final class CPU extends Backend {

    @Override
	public Tensor matmul(Tensor a, Tensor b) {
		return LinearCPU.matmul(a, b);
	}

    @Override
	public void matmul(Tensor a, Tensor b, Tensor dst) {
        LinearCPU.matmul(a, b, dst);
	}

    @Override
	public Tensor corr2D(Tensor x, Tensor k) {
		return ConvCPU.corr2D(x, k);
	}

    @Override
	public void corr2D(Tensor x, Tensor k, Tensor dst) {
        ConvCPU.corr2D(x, k, dst);
	}

	@Override
	public void corr2D(double[] dataX, int offX, double[] dataK, int offK, double[] dataDst, int offDst, int W, int H, int kW, int kH) {
		ConvCPU.corr2D(dataX, offX, dataK, offK, dataDst, offDst, W, H, kW, kH);
	}

    @Override
	public Tensor conv2D(Tensor x, Tensor k) {
		return ConvCPU.conv2D(x, k);
	}

    @Override
	public void conv2D(Tensor x, Tensor k, Tensor dst) {
		ConvCPU.conv2D(x, k, dst);
	}

    @Override
	public Tensor conv2DFull(Tensor x, Tensor k) {
		return ConvCPU.conv2DFull(x, k);
	}

    @Override
	public void conv2DFull(Tensor x, Tensor k, Tensor dst) {
		ConvCPU.conv2DFull(x, k, dst);
	}

	@Override
	public void conv2DFull(
		double[] dataX, int offX,
		double[] dataK, int offK,
		double[] dataDst, int offDst,
		int W, int H,
		int kW, int kH
	) {
		ConvCPU.conv2DFull(dataX, offX, dataK, offK, dataDst, offDst, W, H, kW, kH);
	}

    @Override
	public Tensor maxPool2D(Tensor x, int[] filtro) {
		return PoolingCPU.maxPool2D(x, filtro);
	}

    @Override
	public Tensor maxPool2D(Tensor x, int[] filtro, int[] stride) {
		return PoolingCPU.maxPool2D(x, filtro, stride);
	}

    @Override
	public void maxPool2D(Tensor x, Tensor dst, int[] filtro, int[] stride) {
		PoolingCPU.maxPool2D(x, dst, filtro, stride);
	}

    @Override
	public Tensor avgPool2D(Tensor x, int[] stride) {
		return PoolingCPU.avgPool2D(x, stride);
	}

    @Override
	public Tensor avgPool2D(Tensor x, int[] filtro, int[] stride) {
		return PoolingCPU.avgPool2D(x, filtro, stride);		
	}

    @Override
	public void avgPool2D(Tensor x, Tensor dst, int[] filtro, int[] stride) {
		PoolingCPU.avgPool2D(x, dst, filtro, stride);
	}

	/**
	 * Experimental
	 */
	public Tensor im2col(Tensor x, int kH, int kW) {
		// x: (B, C, H, W)
		int[] shape = x.shape();
		if (shape.length != 4) {
			throw new IllegalArgumentException(
				"im2col espera tensor 4D (B, C, H, W)"
			);
		}

		final int B = shape[0];
		final int C = shape[1];
		final int H = shape[2];
		final int W = shape[3];

		final int outH = H - kH + 1;
		final int outW = W - kW + 1;

		final int lins = B * outH * outW;
		final int cols = C * kH * kW;

		Tensor col = new Tensor(lins, cols);

		final double[] dataX   = x.array();
		final double[] dataCol = col.array();

		final int offXBase   = x.offset();
		final int offColBase = col.offset();
		
		final int areaHW = H * W;
		int lin = 0;
		for (int b = 0; b < B; b++) {
			final int offXB = offXBase + b * C * areaHW;

			for (int oy = 0; oy < outH; oy++) {
				for (int ox = 0; ox < outW; ox++) {

					int colBase = offColBase + lin * cols;
					int idxCol = colBase;

					for (int c = 0; c < C; c++) {
						final int offXC = offXB + c * areaHW;

						for (int ky = 0; ky < kH; ky++) {
							final int inY = oy + ky;
							final int baseIn = offXC + inY * W;

							for (int kx = 0; kx < kW; kx++) {
								dataCol[idxCol++] =
									dataX[baseIn + (ox + kx)];
							}
						}
					}

					lin++;
				}
			}
		}

		return col;
	}

}
