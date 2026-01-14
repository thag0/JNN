package jnn.core.ops;

import jnn.core.tensor.Tensor;
import jnn.nativo.JNNNative;

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

		final double[] dataA = a.array();
		final double[] dataB = b.array();
		final double[] dataD = dst.array();

		for (int i = 0; i < linA; i++) {
			final int baseA = offsetA + i * s0A;
			final int baseD = offsetD + i * s0D;

			for (int k = 0; k < colA; k++) {
				final double valA = dataA[baseA + k * s1A];
				final int baseB = offsetB + k * s0B;

				for (int j = 0; j < colB; j++) {
					dataD[baseD + j * s1D] += valA * dataB[baseB + j * s1B];
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

		final int M = shapeA.length == 1 ? 1 : shapeA[0];
		final int K = shapeA.length == 1 ? shapeA[0] : shapeA[1];
		final int Kb = shapeB.length == 1 ? 1 : shapeB[0];
		final int N = shapeB.length == 1 ? shapeB[0] : shapeB[1];

		if (K != Kb) {
			throw new IllegalArgumentException(
				"Dimensões incompatíveis para multiplicação: A = " + a.shapeStr() + 
				", B = " + b.shapeStr()
			);
		}

		final int linD = shapeD.length == 1 ? 1 : shapeD[0];
		final int colD = shapeD.length == 1 ? shapeD[0] : shapeD[1];

		if (M != linD || N != colD) {
			throw new IllegalArgumentException(
				"Dimensões de saída inválidas, esperado (" + M + "," + N + "), mas recebido " + dst.shapeStr()
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

		JNNNative.matmul(
			a.array(), a.offset(), s0A, s1A,
			b.array(), b.offset(), s0B, s1B,
			dst.array(), dst.offset(), s0C, s1C,
			M, K, N
		);
	}


}
