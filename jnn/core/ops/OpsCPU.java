package jnn.core.ops;

import jnn.core.JNNnative;
import jnn.core.tensor.Tensor;

/**
 * Implementações java em cpu das operações internas da biblioteca.
 */
public final class OpsCPU extends Ops {

	/**
	 * Inicializa um operador focado em cpu.
	 */
	public OpsCPU() {}

    @Override
	public Tensor matmul(Tensor a, Tensor b) {
		return OpsLinear.matmul(a, b);
	}

    @Override
	public void matmul(Tensor a, Tensor b, Tensor dst) {
		if (JNNnative.jni) OpsLinear.matmulJNI(a, b, dst);
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

}
