package jnn.core.tensor;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import jnn.core.OpArray;

/**
 * Auxiliar em operação para tensores 4D.
 */
public class OpTensor4D {

	/**
	 * Operador para tensores 4D.
	 */
	private OpArray oparr = new OpArray();
	
	/**
	 * Auxiliar em operação para tensores 4D.
	 */
	public OpTensor4D() {}

	/**
	 * Verifica se os tensores possuem a mesma altura (dim3) e largura (dim4).
	 * @param a primeiro {@code Tensor}.
	 * @param b segundo {@code Tensor}.
	 * @return resultado da verificação.
	 */
	public boolean compararAlturaLargura(Tensor4D a, Tensor4D b) {
		return (a.dim3() == b.dim3()) && (a.dim4() == b.dim4());
	}

	/**
	 * Copia o conteúdo das duas últimas dimensões do tensor para o destino.
	 * @param tensor {@code Tensor} desejado.
	 * @param dest {@code Tensor} de destino.
	 * @param dimA índies das duas primeiras dimensões do tensor base (dim1, dim2)
	 * @param dimB índies das duas primeiras dimensões do tensor de destino (dim1, dim2)
	 */
	public void copiarMatriz(Tensor4D tensor, Tensor4D dest, int[] dimA, int[] dimB) {
		if (!tensor.comparar2D(dest)) {
			throw new IllegalArgumentException(
				"\nAs duas últimas dimensões do tensor recebido " + tensor.shapeStr() +
				" e de destino " + dest.shapeStr() + " devem ser iguais."
			);
		}

		if (dimA.length != 2 || dimB.length != 2) {
			throw new IllegalArgumentException(
				"\nO array para as dimensões do tensor A e B devem conter dois elementos, recebido " +
				"A = " + dimA.length + ", B = " + dimB.length
			);
		}

		if ((dimA[0] < 0 || dimA[0] >= tensor.dim1()) || (dimA[1] < 0 || dimA[1] >= tensor.dim2())) {
			throw new IllegalArgumentException(
				"\nÍndices do tensor base (" + dimA[0] + ", " + dimA[1] + ") " +
				"inválidos para o tensor com dimensões " + tensor.shapeStr()
			);
		}

		if ((dimB[0] < 0 || dimB[0] >= dest.dim1()) || (dimB[1] < 0 || dimB[1] >= dest.dim2())) {
			throw new IllegalArgumentException(
				"\nÍndices do tensor de destino (" + dimB[0] + ", " + dimB[1] + ") " +
				"inválidos para o tensor de destino com dimensões " + dest.shapeStr()
			);
		}

		for (int i = 0; i < dest.dim3(); i++) {
			for (int j = 0; j < dest.dim4(); j++) {
				dest.set(
					tensor.get(dimA[0], dimA[1], i, j),
					dimB[0], dimB[1], i, j
				);
			}
		}
	}

	/**
	 * Transpõe o conteúdo matricial contido no tensor de acordo com os índices especificados.
	 * @param tensor {@code Tensor} desejado.
	 * @param dim1 índice da primeira dimensão.
	 * @param dim2 índice da segunda dimensão.
	 * @return {@code Tensor} transposto.
	 */
	public Tensor4D matTranspor(Tensor4D tensor, int dim1, int dim2) {
		Tensor4D t = new Tensor4D(1, 1, tensor.dim4(), tensor.dim3());

		for (int i = 0; i < tensor.dim3(); i++) {
			for (int j = 0; j < tensor.dim4(); j++) {
				t.set(
					tensor.get(dim1, dim2, i, j),
					0, 0, j, i
				);
			}
		}

		return t;
	}

	/**
	 * Realiza a adição {@code elemento a elemento} entre matrizes de tensores, como no exemplo.
	 * <pre>
	 *a = [
	 * [[[1, 2], 
	 *   [3, 4]]]  
	 *]
	 *b = [
	 * [[[5, 6], 
	 *   [7, 8]]]  
	 *]
	 *res = [
	 * [[[ 6,  8], 
	 *   [10, 12]]]  
	 *]
	 *
	 * </pre>
	 * Para que a operação aconteça é necessário que {@code todos} os tensores
	 * possuam o mesmo formato.
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @param dest {@code Tensor} de destino do resultado.
	 */
	public void matAdd(Tensor4D a, Tensor4D b, Tensor4D dest) {
		if (!a.comparar4D(b) || !a.comparar4D(dest)) {
			throw new IllegalArgumentException(
				"\nOs tensores devem conter as mesmas dimensões, recebido " +
				"A = " + a.shapeStr() + ", B = " + b.shapeStr() + " e R = " + dest.shapeStr()  
			);
		}

		for (int i = 0; i < a.dim1(); i++) {
			for (int j = 0; j < a.dim2(); j++) {
				
				for (int k = 0; k < a.dim3(); k++) {
					for (int l = 0; l < a.dim4(); l++) {
						dest.set(
							(a.get(i, j, k, l) + b.get(i, j, k, l)),
							i, j, k, l
						);
					}
				}

			}
		}
	}

	/**
	 * Realiza a adição {@code elemento a elemento} entre matrizes de tensores, como no exemplo.
	 * <pre>
	 *    R[i][j] = A[i][j] + B[i][j]
	 * </pre>
	 * @param a primeiro tensor.
	 * @param b segundo tensor.
	 * @param idProf índice da terceira dimensão para os tensores.
	 * @return {@code Tensor} com resultado.
	 */
	public Tensor4D matAdd(Tensor4D a, Tensor4D b, int idProf) {
		int profA = a.dim2();
		int profB = b.dim2();
	
		if ((idProf < 0 || idProf >= profA) || (idProf < 0 || idProf >= profB)) {
			throw new IllegalArgumentException(
				"\nTodos os tensores fornecidos devem conter o índice de profundidade válido."
			);
		}

		if (!compararAlturaLargura(a, b)) {
			throw new IllegalArgumentException(
				"As dimensões de A " + a.shapeStr() + " e B " + b.shapeStr() +
				" devem ser iguais compatíveis"
			);
		}

		int linhas = a.dim3(), colunas = b.dim4();
		Tensor4D res = new Tensor4D(linhas, colunas);
		for (int i = 0; i < linhas; i++) {
			for (int j = 0; j < colunas; j++) {
				res.set(
					(a.get(0, idProf, i, j) + b.get(0, idProf, i, j)),
				 0, idProf, i, j
				);
			}
		}

		return res;
	}

	/**
	 * Realiza a subtração {@code elemento a elemento} entre matrizes de tensores, como no exemplo.
	 * <pre>
	 *a = [
	 * [[[1, 2], 
	 *   [3, 4]]]  
	 *]
	 *b = [
	 * [[[5, 6], 
	 *   [7, 8]]]  
	 *]
	 *res = [
	 * [[[-4, -4], 
	 *   [-4, -4]]]  
	 *]
	 *
	 * </pre>
	 * Para que a operação aconteça é necessário que {@code todos} os tensores
	 * possuam o mesmo formato.
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @param dest {@code Tensor} de destino do resultado.
	 */
	public void matSub(Tensor4D a, Tensor4D b, Tensor4D dest) {
		if (!a.comparar4D(b) || !a.comparar4D(dest)) {
			throw new IllegalArgumentException(
				"\nOs tensores devem conter as mesmas dimensões, recebido " +
				"A = " + a.shapeStr() + ", B = " + b.shapeStr() + " e R = " + dest.shapeStr()  
			);
		}

		for (int i = 0; i < a.dim1(); i++) {
			for (int j = 0; j < a.dim2(); j++) {
				
				for (int k = 0; k < a.dim3(); k++) {
					for (int l = 0; l < a.dim4(); l++) {
						dest.set(
							(a.get(i, j, k, l) - b.get(i, j, k, l)),
							i, j, k, l
						);
					}
				}

			}
		}
	}

	/**
	 * Realiza a subtração {@code elemento a elemento} entre matrizes de tensores, como no exemplo.
	 * <pre>
	 *    R[i][j] = A[i][j] - B[i][j]
	 * </pre>
	 * @param a primeiro tensor.
	 * @param b segundo tensor.
	 * @return tensor com resultado.
	 */
	public Tensor4D matSub(Tensor4D a, Tensor4D b, int idProf) {
		int profA = a.dim2();
		int profB = b.dim2();
	
		if ((idProf < 0 || idProf >= profA) || (idProf < 0 || idProf >= profB)) {
			throw new IllegalArgumentException(
				"\nTodos os tensores fornecidos devem conter o índice de profundidade válido."
			);
		}

		if (!compararAlturaLargura(a, b)) {
			throw new IllegalArgumentException(
				"As dimensões de A " + a.shapeStr() + " e B " + b.shapeStr() +
				" devem ser iguais compatíveis"
			);
		}

		int linhas = a.dim3(), colunas = b.dim4();
		Tensor4D res = new Tensor4D(1, 1, linhas, colunas);
		for (int i = 0; i < linhas; i++) {
			for (int j = 0; j < colunas; j++) {
				res.set(
					(a.get(0, idProf, i, j) - b.get(0, idProf, i, j)), 
				0, idProf, i, j
				);
			}
		}

		return res;
	}

	/**
	 * Realiza multiplicação matricial entre tensores.
	 * <p>
	 *    Para que a multiplicação ocorra é necessário que:
	 * </p>
	 * <ul>
	 *    <li>
	 *       Todos os tensores tenham os mesmos valores para {@code dim1 e dim2};
	 *    </li>
	 *    <li>
	 *       O valor da quarta dimensão de A seja igual a terceira dimensão de B {@code (a.dim4 == b.dim3)};
	 *    </li>
	 *    <li>
	 *       O tensor de saída tenha e terceira e quarta dimensão iguais a {@code (..., ..., a.dim3, b.dim4)}
	 *    </li>
	 * </ul>
	 * Caso nenhum desses critérios seja atendido, é lançada uma exceção.
	 * @param a primeiro {@code Tensor}.
	 * @param b segundo {@code Tensor}.
	 * @param dest {@code Tensor} de destino do resultado.
	 */
	public void matMult(Tensor4D a, Tensor4D b, Tensor4D dest) {
		if (((a.dim1() != b.dim1()) || (a.dim1() != dest.dim1())) ||
			((a.dim2() != b.dim2()) || (a.dim2() != dest.dim2()))) {
			throw new IllegalArgumentException(
				"\nOs tensores A e B devem possuir os mesmos valores para " +
				"as duas primeiras dimensões, recebido A = " 
				+ a.shapeStr() + " e B = " + b.shapeStr() 
			);
		}
		if ((a.dim4() != b.dim3())) {
			throw new IllegalArgumentException(
				"\nA quarta dimensão de A (" + a.dim4() + ") deve" +
				" ser igual a terceira dimensão de B (" + b.dim3() +")"
			);
		}
		if (dest.dim3() != a.dim3() || dest.dim4() != b.dim4()) {
			throw new IllegalArgumentException(
				"\nTensor de resultado contém as duas últimas dimensões inesperadas, " +
				"esperado (" + a.dim3() + ", " + b.dim4() + ")" + " mas possui (" +
				dest.dim3() + ", " + dest.dim4() + ")"
			);
		}

		int rLin = dest.dim3(), rCol = dest.dim4();
		int aCol = a.dim4();
		int aDim1 = a.dim1(), aDim2 = a.dim2();

		//loop pelas primeiras dimensões
		for (int d1 = 0; d1 < aDim1; d1++) {
			for (int d2 = 0; d2 < aDim2; d2++) {
				
				//pela pelas duas últimas dimensões (linhas e colunas)
				for (int i = 0; i < rLin; i++) {
					for (int j = 0; j < rCol; j++) {
						double res = 0;
						for (int k = 0; k < aCol; k++) {
							res += a.get(d1, d2, i, k) * b.get(d1, d2, k, j);
						}
						dest.set(res, d1, d2, i, j);
					}
				}

			}
		}
	}

	/**
	 * Realiza multiplicação {@code elemento a elementos} entre tensores, como no exemplo.
	 * <pre>
	 *a = [
	 * [[[1, 2], 
	 *   [3, 4]]]  
	 *]
	 *b = [
	 * [[[5, 6], 
	 *   [7, 8]]]  
	 *]
	 *res = [
	 * [[[ 5, 12], 
	 *   [21, 32]]]  
	 *]
	 * </pre>
	 * Para que a operação aconteça é necessário que {@code todos} os tensores
	 * possuam o mesmo formato.
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @param dest {@code Tensor} de destino do resultado.
	 */
	public void matHadamard(Tensor4D a, Tensor4D b, Tensor4D dest) {
		if (!a.comparar4D(b) || !a.comparar4D(dest)) {
			throw new IllegalArgumentException(
				"\nOs tensores devem conter as mesmas dimensões, recebido " +
				"A = " + a.shapeStr() + ", B = " + b.shapeStr() + " e R = " + dest.shapeStr()  
			);
		}

		for (int i = 0; i < a.dim1(); i++) {
			for (int j = 0; j < a.dim2(); j++) {
				
				for (int k = 0; k < a.dim3(); k++) {
					for (int l = 0; l < a.dim4(); l++) {
						dest.set(
							(a.get(i, j, k, l) * b.get(i, j, k, l)),
							i, j, k, l
						);
					}
				}
				
			}
		}
	}

	/**
	 * Realiza multiplicação {@code elemento a elementos} entre tensores, como no exemplo.
	 * <pre>
	 *    R[dim1][dim2] = A[dim1][dim2] ⊙ B[dim1][dim2]
	 * </pre>
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @param dim1 índice da primeira dimensão desejada.
	 * @param dim2 índice da segunda dimensão desejada.
	 * @return {@code Tensor} de destino do resultado.
	 */
	public Tensor4D matHadamard(Tensor4D a, Tensor4D b, int dim1, int dim2) {
		int ad1 = a.dim1(), ad2 = a.dim2();
		int bd1 = b.dim1(), bd2 = b.dim2();
	
		if ((dim1 < 0 || dim1 >= ad1) || (dim1 < 0 || dim1 >= bd1)) {
			throw new IllegalArgumentException(
				"\nTodos os tensores fornecidos devem conter o índice de primeira dimensão válido."
			);
		}

		if ((dim1 < 0 || dim1 >= ad2) || (dim1 < 0 || dim1 >= bd2)) {
			throw new IllegalArgumentException(
				"\nTodos os tensores fornecidos devem conter o índice de segunda dimensão válido."
			);
		}
		
		if (!compararAlturaLargura(a, b)) {
			throw new IllegalArgumentException(
				"As duas últimas dimensões de A " + a.shapeStr() + " e B " + b.shapeStr() +
				" devem ser compatíveis"
			);
		}

		Tensor4D res = new Tensor4D(1, 1, a.dim3(), a.dim4());
		int linhas = res.dim3(), colunas = res.dim4();
		for (int i = 0; i < linhas; i++) {
			for (int j = 0; j < colunas; j++) {
				res.set(
					(a.get(dim1, dim2, i, j) * b.get(dim1, dim2, i, j)), 
					dim1, dim2, i, j
				);
			}
		}

		return res;
	}

	/**
	 * Rotaciona em 180° o conteúdo da matriz contido no tensor.
	 * <p>
	 *    Essencialmente esse método é mais para uso de operacões convolucionais.
	 * </p>
	 * @param tensor {@code Tensor} desejado
	 * @param dim1 índice da primeira dimensão do tensor.
	 * @param dim2 índice da segunda dimensão do tensor.
	 * @return {@code Tensor} com uma matriz rotacionada de acordo com os índices dados.
	 */
	public Tensor4D rotacionarMatriz180(Tensor4D tensor, int dim1, int dim2) {
		if (dim2 < 0 || dim2 > tensor.dim1()) {
			throw new IllegalArgumentException(
				"Valor da dimensão 2 (" + dim2 + ") inválido."
			);
		}

		Tensor4D invertido = new Tensor4D(tensor);
		double[] arr = new double[tensor.dim3() * tensor.dim4()];
		
		int cont = 0;
		for (int i = 0; i < tensor.dim3(); i++) {
			for (int j = 0; j < tensor.dim4(); j++) {
				arr[cont] = tensor.get(dim1, dim2, i, j);
				cont++;
			}
		}

		oparr.inverter(arr);
		cont = 0;
		for (int i = 0; i < tensor.dim3(); i++) {
			for (int j = 0; j < tensor.dim4(); j++) {
				invertido.set(arr[cont], dim1, dim2, i, j);
				cont++;
			}
		}

		return invertido;
	}

	/**
	 * Realiza a operação de correlação cruzada (apenas 2D) entre dois tensores usando
	 * a dimensão de profundidade desejada.
	 * <p>
	 *    O resultado será adicionado no tensor de saída de acordo com os índices 
	 *    especificados.
	 * </p>
	 * @param entrada {@code Tensor} com a matriz de entrada.
	 * @param kernel {@code Tensor} com a matriz de kernel.
	 * @param saida {@code Tensor} de destino.
	 * @param idEn índice desejado para a entrada (id[0], id[1] ...).
	 * @param idK índice desejado para o kernel (id[0], id[1] ...).
	 * @param idS índice desejado para a saída (id[0], id[1] ...).
	 */
	public void correlacao2D(Tensor4D entrada, Tensor4D kernel, Tensor4D saida, int[] idEn, int[] idK, int[] idS) {
		if (!entrada.validarDimensao(entrada.dim1(), idEn[0]) || !entrada.validarDimensao(entrada.dim2(), idEn[1])) {
			throw new IllegalArgumentException(
				"\nÍndices de entrada (" + idEn[0] + ", " + idEn[1] + ") " +
				"incompatíveis com o tensor de entrada (" + entrada.dim1() + ", " + entrada.dim2() + ")."
			);
		}
		if (!kernel.validarDimensao(kernel.dim1(), idK[0]) || !kernel.validarDimensao(kernel.dim2(), idK[1])) {
			throw new IllegalArgumentException(
				"\nÍndices do kernel (" + idK[0] + ", " + idK[1] + ") " +
				"incompatíveis com o tensor do kernel (" + kernel.dim1() + ", " + kernel.dim2() + ")."
			);
		}
		if (!saida.validarDimensao(saida.dim1(), idS[0]) || !saida.validarDimensao(saida.dim2(), idS[1])) {
			throw new IllegalArgumentException(
				"\nÍndices da saída (" + idS[0] + ", " + idS[1] + ") " +
				"incompatíveis com o tensor de saída (" + saida.dim1() + ", " + saida.dim2() + ")."
			);
		}

		int altEsperada  = entrada.dim3() - kernel.dim3() + 1;
		int largEsperada = entrada.dim4() - kernel.dim4() + 1;
		if (saida.dim3() != altEsperada || saida.dim4() != largEsperada) {
			throw new IllegalArgumentException(
				"\nDimensões de saída inconpatíveis, esperado (" + altEsperada + ", " + largEsperada + ")" +
				", recebido (" + saida.dim3() + ", " + saida.dim4() + ")."
			);
		}

		//cache
		double[][] entradaLocal = entrada.array2D(idEn[0], idEn[1]);
		double[][] kernelLocal = kernel.array2D(idK[0], idK[1]);

		int alturaKernel = kernel.dim3();
		int larguraKernel = kernel.dim4();
		for (int i = 0; i < altEsperada; i++) {
			for (int j = 0; j < largEsperada; j++) {
				
				double soma = 0.0;
				for (int k = 0; k < alturaKernel; k++) {
					int posX = i + k;
					for (int l = 0; l < larguraKernel; l++) {
						soma += entradaLocal[posX][j+l] * kernelLocal[k][l];
					}
				}

				saida.add(idS[0], idS[1], i, j, soma);
			}
		}
	}

	/**
	 * Realiza a operação de correlação cruzada (apenas 2D) entre dois tensores usando
	 * a dimensão primeira dimensão dos tensores e a segunda dimensão (profundidade) desejada.
	 * <p>
	 *    O resultado será adicionado no tensor de saída de acordo com os índices 
	 *    especificados.
	 * </p>
	 * @param entrada {@code Tensor} com a matriz de entrada.
	 * @param kernel {@code Tensor} com a matriz de kernel.
	 * @param saida {@code Tensor} de destino.
	 * @param idProf índice do canal de profundidade desejado.
	 */
	public void correlacao2D(Tensor4D entrada, Tensor4D kernel, Tensor4D saida, int idProf) {
		correlacao2D(
			entrada, kernel, saida, 
			new int[]{0, idProf}, 
			new int[]{0, idProf}, 
			new int[]{0, idProf}
		);
	}

	/**
	 * Realiza a operação de correlação cruzada (apenas 2D) entre dois tensores usando
	 * as duas primeiras dimensões de profundidade.
	 * <p>
	 *    O resultado será adicionado no tensor de saída de acordo com os índices 
	 *    especificados.
	 * </p>
	 * @param entrada {@code Tensor} com a matriz de entrada.
	 * @param kernel {@code Tensor} com a matriz de kernel.
	 * @param saida {@code Tensor} de destino.
	 */
	public void correlacao2D(Tensor4D entrada, Tensor4D kernel, Tensor4D saida) {
		correlacao2D(
			entrada, kernel, saida, 
			new int[]{0, 0}, 
			new int[]{0, 0}, 
			new int[]{0, 0}
		);
	}

	/**
	 * Realiza a operação de convolução (apenas 2D) entre dois tensores usando
	 * a dimensão de profundidade desejada.
	 * <p>
	 *    O resultado será adicionado no tensor de saída de acordo com os índices 
	 *    especificados.
	 * </p>
	 * @param entrada {@code Tensor} com a matriz de entrada.
	 * @param kernel {@code Tensor} com a matriz de kernel.
	 * @param saida {@code Tensor} de destino.
	 * @param idEn índice desejado para a entrada (id[0], id[1] ...).
	 * @param idK índice desejado para o kernel (id[0], id[1] ...).
	 * @param idS índice desejado para a saída (id[0], id[1] ...).
	 */
	public void convolucao2D(Tensor4D entrada, Tensor4D kernel, Tensor4D saida, int[] idEn, int[] idK, int[] idS) {
		if (!entrada.validarDimensao(entrada.dim1(), idEn[0]) || !entrada.validarDimensao(entrada.dim2(), idEn[1])) {
			throw new IllegalArgumentException(
				"\nÍndices de entrada (" + idEn[0] + ", " + idEn[1] + ") " +
				"incompatíveis com o tensor de entrada (" + entrada.dim1() + ", " + entrada.dim2() + ")."
			);
		}
		if (!kernel.validarDimensao(kernel.dim1(), idK[0]) || !kernel.validarDimensao(kernel.dim2(), idK[1])) {
			throw new IllegalArgumentException(
				"\nÍndices do kernel (" + idK[0] + ", " + idK[1] + ") " +
				"incompatíveis com o tensor do kernel (" + kernel.dim1() + ", " + kernel.dim2() + ")."
			);
		}
		if (!saida.validarDimensao(saida.dim1(), idS[0]) || !saida.validarDimensao(saida.dim2(), idS[1])) {
			throw new IllegalArgumentException(
				"\nÍndices da saída (" + idS[0] + ", " + idS[1] + ") " +
				"incompatíveis com o tensor de saída (" + saida.dim1() + ", " + saida.dim2() + ")."
			);
		}

		int alturaEsperada  = entrada.dim3() - kernel.dim3() + 1;
		int larguraEsperada = entrada.dim4() - kernel.dim4() + 1;
		if (saida.dim3() != alturaEsperada) {
			throw new IllegalArgumentException(
				"\nAltura da saída (" + saida.dim3() + 
				") íncompatível com o valor esperado (" + alturaEsperada + ")."
			);
		}
		if (saida.dim4() != larguraEsperada) {
			throw new IllegalArgumentException(
				"\nAltura da saída (" + saida.dim4() + 
				") íncompatível com o valor esperado (" + larguraEsperada + ")."
			);
		}

		Tensor4D rotacionado = rotacionarMatriz180(kernel, idK[0], idK[1]);

		int alturaKernel = rotacionado.dim3();
		int larguraKernel = rotacionado.dim4();
		int i, j, m, n;
		double soma;
		int posX, posY;
		for (i = 0; i < alturaEsperada; i++) {
			for (j = 0; j < larguraEsperada; j++) {

				soma = 0.0;
				for (m = 0; m < alturaKernel; m++) {
					posX = i + m;
					for (n = 0; n < larguraKernel; n++) {
						posY = j + n;
						soma += entrada.get(idEn[0], idEn[1], posX, posY) * 
								rotacionado.get(idK[0], idK[1], m, n);
					}
				}

				saida.add(idS[0], idS[1], i, j, soma);
			}
		}
	}

	/**
	 * Realiza a operação de convolução (apenas 2D) entre dois tensores usando
	 * a dimensão primeira dimensão dos tensores e a segunda dimensão (profundidade) desejada.
	 * <p>
	 *    O resultado será adicionado no tensor de saída de acordo com os índices 
	 *    especificados.
	 * </p>
	 * @param entrada {@code Tensor} com a matriz de entrada.
	 * @param kernel {@code Tensor} com a matriz de kernel.
	 * @param saida {@code Tensor} de destino.
	 * @param idProf índice do canal de profundidade desejado.
	 */
	public void convolucao2D(Tensor4D entrada, Tensor4D kernel, Tensor4D saida, int idProf) {
		convolucao2D(
			entrada, kernel, saida, 
			new int[]{0, idProf}, 
			new int[]{0, idProf}, 
			new int[]{0, idProf}
		);
	}

	/**
	 * Realiza a operação de correlação convolução (apenas 2D) entre dois tensores usando
	 * a primeira dimensão de profundidade.
	 * <p>
	 *    O resultado será adicionado no tensor de saída de acordo com os índices 
	 *    especificados.
	 * </p>
	 * @param entrada {@code Tensor} com a matriz de entrada.
	 * @param kernel {@code Tensor} com a matriz de kernel.
	 * @param saida tensor de destino.
	 */
	public void convolucao2D(Tensor4D entrada, Tensor4D kernel, Tensor4D saida) {
		convolucao2D(
			entrada, kernel, saida, 
			new int[]{0, 0}, 
			new int[]{0, 0}, 
			new int[]{0, 0}
		);
	}

	/**
	 * Realiza a operação de convolução (apenas 2D) entre dois tensores usando
	 * a dimensão de profundidade desejada.
	 * <p>
	 *    O resultado será adicionado no tensor de saída de acordo com os índices 
	 *    especificados.
	 * </p>
	 * @param entrada {@code Tensor} com a matriz de entrada.
	 * @param kernel {@code Tensor} com a matriz de kernel.
	 * @param saida {@code Tensor} de destino.
	 * @param idEn índice desejado para a entrada (id[0], id[1] ...).
	 * @param idK índice desejado para o kernel (id[0], id[1] ...).
	 * @param idS índice desejado para a saída (id[0], id[1] ...).
	 */
	public void convolucao2DFull(Tensor4D entrada, Tensor4D kernel, Tensor4D saida, int[] idEn, int[] idK, int[] idS) {
		if (!entrada.validarDimensao(entrada.dim1(), idEn[0]) || !entrada.validarDimensao(entrada.dim2(), idEn[1])) {
			throw new IllegalArgumentException(
				"\nÍndices de entrada (" + idEn[0] + ", " + idEn[1] + ") " +
				"incompatíveis com o tensor de entrada (" + entrada.dim1() + ", " + entrada.dim2() + ")."
			);
		}
		if (!kernel.validarDimensao(kernel.dim1(), idK[0]) || !kernel.validarDimensao(kernel.dim2(), idK[1])) {
			throw new IllegalArgumentException(
				"\nÍndices do kernel (" + idK[0] + ", " + idK[1] + ") " +
				"incompatíveis com o tensor do kernel (" + kernel.dim1() + ", " + kernel.dim2() + ")."
			);
		}
		if (!saida.validarDimensao(saida.dim1(), idS[0]) || !saida.validarDimensao(saida.dim2(), idS[1])) {
			throw new IllegalArgumentException(
				"\nÍndices da saída (" + idS[0] + ", " + idS[1] + ") " +
				"incompatíveis com o tensor de saída (" + saida.dim1() + ", " + saida.dim2() + ")."
			);
		}

		int altEsp  = entrada.dim3() + kernel.dim3() - 1;
		int largEsp = entrada.dim4() + kernel.dim4() - 1;
		if (saida.dim3() != altEsp || saida.dim4() != largEsp) {
			throw new IllegalArgumentException(
				"\nDimensões de saída inconpatíveis, esperado (" + altEsp + ", " + largEsp + ")" +
				", recebido (" + saida.dim3() + ", " + saida.dim4() + ")."
			);
		}

		int linEntrada = entrada.dim3(), colEntrada = entrada.dim4();
		int linKernel = kernel.dim3(), colKernel = kernel.dim4();
		int linSaida = saida.dim3(), colSaida = saida.dim4();

		//cache
		double[][] entradaLocal = entrada.array2D(idEn[0], idEn[1]);
		double[][] kernelLocal = kernel.array2D(idK[0], idK[1]);
		
		int posX, posY;
		double soma;
		int i, j, k, l;
		for (i = 0; i < linSaida; i++) {
			for (j = 0; j < colSaida; j++) {
				
				soma = 0;
				for (k = 0; k < linKernel; k++) {
					posX = i - k;
					if (posX >= 0 && posX < linEntrada) {
						for (l = 0; l < colKernel; l++) {
							posY = j - l;
							if (posY >= 0 && posY < colEntrada) {
								soma += entradaLocal[posX][posY] * kernelLocal[k][l];
							}
						}
					} 
				}
				
				saida.add(idS[0], idS[1], i, j, soma);
			}
		}
	}

	/**
	 * Método exclusivo para a propagação direta de camadas convolucionais
	 * @param entrada {@code Tensor} de entrada.
	 * @param kernel {@code Tensor} dos kernels.
	 * @param saida {@code Tensor} de destino.
	 */
	public void convForward(Tensor4D entrada, Tensor4D kernel, Tensor4D saida) {
		int filtros = kernel.dim1();
		int entradas = kernel.dim2();

		int[] idSaida = {0, 0};
		int[] idEntrada = {0, 0};
		int[] idKernel = {0, 0};

		for (int i = 0; i < filtros; i++) {
			idSaida[1] = i;
			idKernel[0] = i;
			for (int j = 0; j < entradas; j++) {
				idEntrada[1] = j;
				idKernel[1] = j;
				correlacao2D(entrada, kernel, saida, idEntrada, idKernel, idSaida);
			}
		}
	}
	
	/**
	 * Método exclusivo para a propagação reversa de camadas convolucionais.
	 * @param entrada {@code Tensor} de entrada da camada.
	 * @param kernel {@code Tensor} dos kernels.
	 * @param gradSaida {@code Tensor} com os valores dos gradientes da camada em relação a sua saída.
	 * @param gradK {@code Tensor} dos gradientes em relação aos filtros.
	 * @param gradE {@code Tensor} com o gradiente de entrada.
	 */
	public void convBackward(Tensor4D entrada, Tensor4D kernel, Tensor4D gradSaida, Tensor4D gradK, Tensor4D gradE) {
		final int filtros = kernel.dim1();
		final int entradas = kernel.dim2();

		// NOTA
		//essa solução não é definitiva ainda porque nos testes não teve
		//uma grande melhora, apenas cerca de 5~6% de melhoria no desempenho
		//treinando com modelos pequenos e poucas épocas
		ExecutorService exec = Executors.newFixedThreadPool(1);

		exec.submit(() -> {
			for (int i = 0; i < filtros; i++) {
				for (int j = 0; j < entradas; j++) {
					convolucao2DFull(gradSaida, kernel, gradE, new int[]{0, i}, new int[]{i, j}, new int[]{0, j});
				}
			}
		});
		exec.shutdown();

		for (int i = 0; i < filtros; i++) {
			for (int j = 0; j < entradas; j++) {
				correlacao2D(entrada, gradSaida, gradK, new int[]{0, j}, new int[]{0, i}, new int[]{i, j});
			}
		}

		while (!exec.isTerminated()) {}
	}
}
