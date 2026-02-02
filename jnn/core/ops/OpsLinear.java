package jnn.core.ops;

import jnn.core.JNNnative;
import jnn.core.tensor.Tensor;

/**
 * 
 */
public class OpsLinear {
    
    public static Tensor matadd(Tensor a, Tensor b) {
		if (!a.compShape(b)) {
			throw new IllegalArgumentException(
				"\nDimensões do tensor A " + a.shapeStr() + 
				" e B " + b.shapeStr() + " devem ser iguais."
			);
		}

		return a.map(b, (x, y) -> x + y);
    }

	public static Tensor matsub(Tensor a, Tensor b) {
		if (!a.compShape(b)) {
			throw new IllegalArgumentException(
				"\nDimensões do tensor A " + a.shapeStr() + 
				" e B " + b.shapeStr() + " devem ser iguais."
			);
		}

		return a.map(b, (x, y) -> x - y);
	}

	public static Tensor mathad(Tensor a, Tensor b) {
		if (!a.compShape(b)) {
			throw new IllegalArgumentException(
				"\nDimensões do tensor A " + a.shapeStr() + 
				" e B " + b.shapeStr() + " devem ser iguais."
			);
		}

		return a.map(b, (x, y) -> x * y);
	}

	public static Tensor matdiv(Tensor a, Tensor b) {
		if (!a.compShape(b)) {
			throw new IllegalArgumentException(
				"\nDimensões do tensor A " + a.shapeStr() + 
				" e B " + b.shapeStr() + " devem ser iguais."
			);
		}
		return a.map(b, (x, y) -> x / y);
	}

	public static Tensor matmul(Tensor a, Tensor b) {
		if (a.numDim() > 2 || b.numDim() > 2) {
			throw new IllegalArgumentException(
				"\nOs tensores devem conter até duas dimensões, mas contêm " +
				"A = " + a.numDim() + " B = " + b.numDim()
			);
		}
	
		int[] shapeA = a.shape();
		int[] shapeB = b.shape();

		int linA = shapeA.length == 1 ? 1 : shapeA[0];
		int colA = shapeA.length == 1 ? shapeA[0] : shapeA[1];
		int linB = shapeB.length == 1 ? 1 : shapeB[0];
		int colB = shapeB.length == 1 ? shapeB[0] : shapeB[1];
	
		if (colA != linB) {
			throw new IllegalArgumentException(
				"As dimensões dos tensores não são compatíveis para multiplicação de matrizes: " +
				"A = " + a.shapeStr() + " B = " + b.shapeStr()
			);
		}
	
		Tensor res = linA == 1 ? new Tensor(colB) : new Tensor(linA, colB);
		
		matmul(a, b, res);
	
		return res;
	}

	public static void matmul(Tensor a, Tensor b, Tensor dst) {
		if (a.numDim() > 2 || b.numDim() > 2 || dst.numDim() > 2) {
			throw new IllegalArgumentException(
				"\nOs tensores devem conter até duas dimensões, mas contêm " +
				"A = " + a.numDim() + " B = " + b.numDim() + " Dest = " + dst.numDim()
			);
		}

		final int[] shapeA = a.shape();
		final int[] shapeB = b.shape();
		final int[] shapeD = dst.shape();

		final int linA = shapeA.length == 1 ? 1 : shapeA[0];
		final int colA = shapeA.length == 1 ? shapeA[0] : shapeA[1];
		final int linB = shapeB.length == 1 ? 1 : shapeB[0];
		final int colB = shapeB.length == 1 ? shapeB[0] : shapeB[1];
		final int linD = shapeD.length == 1 ? 1 : shapeD[0];
		final int colD = shapeD.length == 1 ? shapeD[0] : shapeD[1];
	
		if (colA != linB) {
			throw new IllegalArgumentException(
				"As dimensões dos tensores não são compatíveis para multiplicação de matrizes: " +
				"A = " + a.shapeStr() + " B = " + b.shapeStr()
			);
		}

		if (linA != linD || colB != colD) {
			throw new IllegalArgumentException(
				"\nDimensões de saída inesperadas, esperado (" + linA + ", " + colB +  ")" +
				", mas recebido " + dst.shapeStr() 
			);
		}
	
		final int[] stridesA = a.strides();
		final int[] stridesB = b.strides();
		final int[] stridesD = dst.strides();

		final int s0A = stridesA.length == 1 ? 1 : stridesA[0];
		final int s1A = stridesA.length == 1 ? 1 : stridesA[1];
		final int s0B = stridesB.length == 1 ? 1 : stridesB[0];
		final int s1B = stridesB.length == 1 ? 1 : stridesB[1];
		final int s0D = stridesD.length == 1 ? 1 : stridesD[0];
		final int s1D = stridesD.length == 1 ? 1 : stridesD[1];

		final int offsetA = a.offset();
		final int offsetB = b.offset();
		final int offsetD = dst.offset();

		final float[] dataA = a.array();
		final float[] dataB = b.array();
		final float[] dataD = dst.array();

		if (s1A == 1 && s1B == 1 && s1D == 1) {// tensores contiguos
			matmulFastPath(
				dataA, dataB, dataD, offsetA, offsetB, offsetD, linA, colA, colB, s0A, s0B, s0D
			);
		} else {
			matmulGenerico(
				dataA, dataB, dataD, offsetA, offsetB, offsetD, linA, colA, colB, s0A, s1A, s0B, s1B, s0D, s1D
			);
		}

	}

	private static void matmulFastPath(
		float[] A, float[] B, float[] C, 
		int offA, int offB, int offC,
		int linA, int colA, int colB,
		int s0A, int s0B, int s0C) {

		final int BK = 64;
		final int BJ = 64;

		for (int i = 0; i < linA; i++) {
			final int baseA = offA + i * s0A;
			final int baseC = offC + i * s0C;

			for (int kk = 0; kk < colA; kk += BK) {
				final int kEnd = Math.min(kk + BK, colA);

				for (int jj = 0; jj < colB; jj += BJ) {
					final int jEnd = Math.min(jj + BJ, colB);

					for (int k = kk; k < kEnd; k++) {
						final float valA = A[baseA + k];
						final int baseB = offB + k * s0B;

						for (int j = jj; j < jEnd; j++) {
							C[baseC + j] += valA * B[baseB + j];
						}
					}
				}
			}
		}
	}

	private static void matmulGenerico(
		float[] A, float[] B, float[] C,
		int offA, int offB, int offC,
		int linA, int colA, int colB,
		int s0A, int s1A,
		int s0B, int s1B,
		int s0C, int s1C) {

		final int BK = 64;
		final int BJ = 64;

		for (int i = 0; i < linA; i++) {
			final int baseA = offA + i * s0A;
			final int baseC = offC + i * s0C;

			for (int kk = 0; kk < colA; kk += BK) {
				final int kEnd = Math.min(kk + BK, colA);

				for (int jj = 0; jj < colB; jj += BJ) {
					final int jEnd = Math.min(jj + BJ, colB);

					for (int k = kk; k < kEnd; k++) {
						final float valA = A[baseA + k * s1A];
						final int baseB = offB + k * s0B;

						for (int j = jj; j < jEnd; j++) {
							C[baseC + j * s1C] += valA * B[baseB + j * s1B];
						}
					}
				}
			}
		}
	}

	public static void matmul_jni(Tensor a, Tensor b, Tensor dst) {
		if (a.numDim() > 2 || b.numDim() > 2 || dst.numDim() > 2)
			throw new IllegalArgumentException(
				"\nOs tensores devem conter até duas dimensões, mas contêm " +
				"A = " + a.numDim() + " B = " + b.numDim() + " Dest = " + dst.numDim()
			);

		final int[] shapeA = a.shape();
		final int[] shapeB = b.shape();
		final int[] shapeD = dst.shape();

		final int linA = shapeA.length == 1 ? 1 : shapeA[0];
		final int colA = shapeA.length == 1 ? shapeA[0] : shapeA[1];
		final int linB = shapeB.length == 1 ? 1 : shapeB[0];
		final int colB = shapeB.length == 1 ? shapeB[0] : shapeB[1];

		if (colA != linB) {
			throw new IllegalArgumentException(
				"Dimensões incompatíveis para multiplicação: A = " + a.shapeStr() + 
				", B = " + b.shapeStr()
			);
		}

		final int linD = shapeD.length == 1 ? 1 : shapeD[0];
		final int colD = shapeD.length == 1 ? shapeD[0] : shapeD[1];

		if (linA != linD || colB != colD) {
			throw new IllegalArgumentException(
				"Dimensões de saída inválidas, esperado (" + linA + "," + colB + "), mas recebido " + dst.shapeStr()
			);
		}

		final int[] stridesA = a.strides();
		final int[] stridesB = b.strides();
		final int[] stridesD = dst.strides();

		final int s0A = stridesA.length == 1 ? 1 : stridesA[0];
		final int s1A = stridesA.length == 1 ? 1 : stridesA[1];
		final int s0B = stridesB.length == 1 ? 1 : stridesB[0];
		final int s1B = stridesB.length == 1 ? 1 : stridesB[1];
		final int s0C = stridesD.length == 1 ? 1 : stridesD[0];
		final int s1C = stridesD.length == 1 ? 1 : stridesD[1];

		JNNnative.matmul(
			a.array(), a.offset(), s0A, s1A,
			b.array(), b.offset(), s0B, s1B,
			dst.array(), dst.offset(), s0C, s1C,
			linA, colA, colB
		);
	}


}
