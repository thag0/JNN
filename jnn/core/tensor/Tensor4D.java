package jnn.core.tensor;

import java.util.Iterator;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * <h2>
 *		Tensor quadridimensional
 * </h2>
 * Implementação de um array multidimensional (atualmente com no máximo quatro
 * dimentões) com finalidade de simplificar o uso de estrutura de dados dentro
 * da biblioteca.
 * <p>
 * 		O tensor possui algumas funções próprias com intuito de aproveitar a
 * 		velocidade
 * 		de processamento usando um único array contendo os dados do dele.
 * </p>
 * <p>
 * 		Algumas operações mais elaboradas podem precisar do auxílio da classe
 * 		{@code OpTensor4D},
 * 		que implementa operações entre vários tensores.
 * </p>
 * <h2>
 *		Exemplo de criação:
 * </h2>
 * <pre>
 *Tensor4D tensor = new Tensor4D(1, 1, 2, 2);
 *Tensor4D tensor = new Tensor4D(new int[]{2, 2});
 *Tensor4D tensor = new Tensor4D(2, 2);
 *tensor = [
 *  [[[0.0, 0.0],
 *    [0.0, 0.0]]]
 *]
 * </pre>
 * Algumas operações entre tensores são válidas desde que as dimensões
 * de ambos os tensores sejam iguais.
 * <pre>
 *Tensor4D a = new Tensor4D(1, 1, 2, 2);
 *a.preencer(1);
 *Tensor4D b = new Tensor4D(1, 1, 2, 2);
 *b.preencer(2);
 *a.add(b);//operação acontece dentro do tensor A
 *a = [
 *  [[[3.0, 3.0],
 *    [3.0, 3.0]]]
 *]
 * </pre>
 * 
 * @author Thiago Barroso, acadêmico de Engenharia da Computação pela
 * Universidade Federal do Pará, Campus Tucuruí. Fevereiro/2023.
 */
public class Tensor4D implements Cloneable, Iterable<Double> {

	/**
	 * Dimensões do tensor (d1, d2, d3, d4).
	 */
	private final int[] shape;

	/**
	 * Conjunto de elementos do tensor.
	 */
	private final double[] dados;

	/**
	 * Nome do tensor.
	 */
	private String nome = getClass().getSimpleName();

	/**
	 * Inicializa um tensor com quatro dimensões a partir de outra instância de
	 * Tensor4D.
	 * <p>
	 *		O conteúdo do tensor recebido será copiado.
	 * </p>
	 * @param tensor tensor desejado.
	 */
	public Tensor4D(Tensor4D tensor) {
		if (tensor == null) {
			throw new IllegalArgumentException(
				"O tensor fornecido é nulo."
			);
		}

		this.shape = new int[4];
		copiarDimensoes(tensor.shape());
		this.dados = new double[tensor.tamanho()];

		System.arraycopy(tensor.dados, 0, this.dados, 0, tamanho());
	}

	/**
	 * Inicializa um tensor com quatro dimensões a partir de um tensor
	 * quadridimensional
	 * primitivo.
	 * <p>
	 *		O formato do tensor criado será:
	 * </p>
	 * <pre>
	 *    formato = (dim1, dim2, dim3, dim4)
	 * </pre>
	 * @param tensor tensor desejado.
	 */
	public Tensor4D(double[][][][] tensor) {
		if (tensor == null) {
			throw new IllegalArgumentException(
				"\nO tensor fornecido é nulo."
			);
		}

		this.shape = new int[4];
		copiarDimensoes(
			tensor.length,
			tensor[0].length,
			tensor[0][0].length,
			tensor[0][0][0].length
		);

		this.dados = new double[dim1() * dim2() * dim3() * dim4()];

		copiar(tensor);
	}

	/**
	 * Inicializa um tensor com quatro dimensões a partir de um tensor
	 * tridimensional
	 * primitivo.
	 * <p>
	 * O formato do tensor criado será:
	 * </p>
	 * <pre>
	 *    formato = (1, profundidade, linhas, colunas)
	 * </pre>
	 * @param tensor tensor desejado.
	 */
	public Tensor4D(double[][][] tensor) {
		if (tensor == null) {
			throw new IllegalArgumentException(
				"\nO tensor fornecido é nulo."
			);
		}

		this.shape = new int[4];
		copiarDimensoes(tensor.length, tensor[0].length, tensor[0][0].length);
		this.dados = new double[dim1() * dim2() * dim3() * dim4()];

		copiar(tensor, 0);
	}

	/**
	 * Inicializa um tensor com quatro dimensões a partir de uma matriz primitiva.
	 * <p>
	 * O formato do tensor criado será:
	 * </p>
	 * <pre>
	 *    formato = (1, 1, linhas, colunas)
	 * </pre>
	 * @param matriz matriz desejado.
	 */
	public Tensor4D(double[][] matriz) {
		if (matriz == null) {
			throw new IllegalArgumentException(
				"\nA matriz fornecida é nula."
			);
		}

		int col = matriz[0].length;
		for (int i = 1; i < matriz.length; i++) {
			if (matriz[i].length != col) {
				throw new IllegalArgumentException(
					"\nA matriz deve conter a mesma quantidade de linhas para todas as colunas."
				);
			}
		}

		this.shape = new int[4];
		copiarDimensoes(matriz.length, matriz[0].length);
		this.dados = new double[dim1() * dim2() * dim3() * dim4()];

		copiar(matriz, 0, 0);
	}

	/**
	 * Inicializa um tensor com quatro dimensões a partir de um array primitivo.
	 * <p>
	 * O formato do tensor criado será:
	 * </p>
	 * <pre>
	 *    formato = (1, 1, 1, tamanhoArray)
	 * </pre>
	 * @param array array desejado.
	 */
	public Tensor4D(double[] array) {
		this(1, 1, 1, array.length, array);
	}

	/**
	 * Inicializa um tensor com quatro dimensões de acordo com os valores
	 * fornecidos.
	 * @param dim1 índice da primeira dimensão.
	 * @param dim2 índice da segunda dimensão.
	 * @param dim3 índice da terceira dimensão.
	 * @param dim4 índice da quarta dimensão.
	 * @param elementos array de elementos com dados predefinidos.
	 */
	public Tensor4D(int dim1, int dim2, int dim3, int dim4, double[] elementos) {
		if (dim1 < 1 || dim2 < 1 || dim3 < 1 || dim4 < 1) {
			throw new IllegalArgumentException(
				"\nOs valores de dimensões não podem ser menores que 1, recebido: (" +
				dim1 + ", " + dim2 + ", " + dim3 + ", " + dim4 + ")."
			);
		}

		if (elementos == null) {
			throw new IllegalArgumentException(
				"\nArray fornecido é nulo."
			);
		}

		if ((dim1 * dim2 * dim3 * dim4) != elementos.length) {
			throw new IllegalArgumentException(
				"\nOs valores de índices não correspondem a quantidade de elementos recebida."
			);
		}

		this.shape = new int[4];
		copiarDimensoes(dim1, dim2, dim3, dim4);
		this.dados = new double[elementos.length];
		System.arraycopy(elementos, 0, dados, 0, dados.length);
	}

	/**
	 * Inicializa um tensor com quatro dimensões de acordo com os valores
	 * fornecidos.
	 * <p>
	 * O conteúdo do tensor estará zerado.
	 * </p>
	 * @param dim1 índice da primeira dimensão.
	 * @param dim2 índice da segunda dimensão.
	 * @param dim3 índice da terceira dimensão.
	 * @param dim4 índice da quarta dimensão.
	 */
	public Tensor4D(int dim1, int dim2, int dim3, int dim4) {
		this(dim1, dim2, dim3, dim4, new double[dim1 * dim2 * dim3 * dim4]);
	}

	/**
	 * Inicializa um tensor com quatro dimensões de acordo com os valores
	 * fornecidos.
	 * <p>
	 * O conteúdo do tensor estará zerado.
	 * </p>
	 * @param dim array de dimensões contendo os valores em ordem crescente
	 * das dimensões do tensor (d1, d2, d3, d4)
	 */
	public Tensor4D(int... dim) {
		if (dim.length == 0) {
			throw new IllegalArgumentException(
				"\nA quantidade de dimensões deve ser maior que zero."
			);
		}

		if (dim.length > 4) {
			throw new IllegalArgumentException(
				"\nA quantidade de dimensões deve conter no máximo quatro elementos, " +
				"recebido " + dim.length + "."
			);
		}

		for (int i = 0; i < dim.length; i++) {
			if (dim[i] < 1) {
				throw new IllegalArgumentException(
					"\nOs valores de dimensões devem ser maiores que zero."
				);
			}
		}

		this.shape = new int[4];
		copiarDimensoes(dim);
		this.dados = new double[dim1() * dim2() * dim3() * dim4()];
	}

	/**
	 * Auxliar para cópia de um array de dimensões para as dimensões do tensor.
	 * @param dim array de dimensões.
	 */
	private void copiarDimensoes(int... dim) {
		int n1 = this.shape.length;
		for (int i = 0; i < n1; i++) {
			this.shape[i] = 1;
		}

		int n2 = dim.length;
		for (int i = 0; i < n2; i++) {
			this.shape[n1 - 1 - i] = dim[n2 - 1 - i];
		}
	}

	/**
	 * Configura o novo formato para o tensor.
	 * <p>
	 * A configuração não altera o conteúdo do tensor, e sim a forma
	 * como os dados são tratados e acessados.
	 * </p>
	 * Exemplo:
	 * <pre>
	 *tensor = [
	 *  [
	 *    [[1, 2],
	 *     [3, 4]]
	 *  ]  
	 *]
	 *
	 *int[] novoFormato = {1, 1, 1, 4};
	 *tensor.reformatar(novoFormato);
	 *
	 *tensor = [
	 *  [
	 *    [[1, 2, 3, 4]]
	 *  ]  
	 *]
	 * </pre>
	 * @param dim array contendo as novas dimensões (dim1, dim2, dim3, dim4).
	 * @return instância local alterada.
	 */
	public Tensor4D reshape(int... dim) {
		if (dim.length == 0 || dim.length > 4) {
			throw new IllegalArgumentException(
				"\nQuantidade de índices deve ser de no máximo 4, recebido " +
				dim.length + "."
			);
		}

		// novas dimensões
		int[] dims = { 1, 1, 1, 1 };

		int tam1 = dims.length;
		int tam2 = dim.length;
		for (int i = 0; i < tam2; i++) {
			dims[tam1 - 1 - i] = dim[tam2 - 1 - i];
		}

		if ((dims[0] < 1) || (dims[1] < 1) || (dims[2] < 1) || (dims[3] < 1)) {
			throw new IllegalArgumentException(
				"\nOs novos valores de dimensões devem ser maiores que zero."
			);
		}

		if ((dims[0] * dims[1] * dims[2] * dims[3]) != tamanho()) {
			throw new IllegalArgumentException(
				"\nA quatidade de elementos com as novas dimensões (" + (dims[0] * dims[1] * dims[2] * dims[3]) +
				") deve ser igual a quantidade de elementos do tensor (" + tamanho() + ")."
			);
		}

		copiarDimensoes(dims[0], dims[1], dims[2], dims[3]);

		return this;
	}

	/**
	 * Calcula o índice do elemento dentro do array de elementos do tensor.
	 * @param dim1 índice da primeira dimensão.
	 * @param dim2 índice da segunda dimensão.
	 * @param dim3 índice da terceira dimensão.
	 * @param dim4 índice da quarta dimensão.
	 * @return índice calculado.
	 */
	private int indice(int dim1, int dim2, int dim3, int dim4) {
		return  (dim1 * dim2() * dim3() * dim4()) +
				(dim2 * dim3() * dim4()) +
				(dim3 * dim4()) +
				 dim4;
	}

	/**
	 * Retorna o elemento do tensor de acordo com os índices fornecidos.
	 * <p>
	 * O formato dos índices deve se dar por: {@code (dim1, dim2, dim3, dim4)}
	 * </p>
	 * @param indices índices desejados para busca.
	 * @return valor de acordo com os índices.
	 */
	public double get(int... indices) {
		switch (indices.length) {
			case 1:
				return dados[indice(0, 0, 0, indices[0])];
			
			case 2:
				return dados[indice(0, 0, indices[0], indices[1])];
			
			case 3:
				return dados[indice(0, indices[0], indices[1], indices[2])];
			
			case 4:
				return dados[indice(indices[0], indices[1], indices[2], indices[3])];
		
			default:
				throw new IndexOutOfBoundsException(
					STR."\nQuantidade de índices (\{indices.length}) inválida"
				);
		}
	}

	/**
	 * Edita o valor do tensor usando o valor informado.
	 * <p>
	 * O formato dos índices deve se dar por: {@code (dim1, dim2, dim3, dim4)}
	 * </p>
	 * @param indices índices para atribuição.
	 * @param valor valor desejado.
	 */
	public void set(double valor, int... indices) {
		switch (indices.length) {
			case 1:
				dados[indice(0, 0, 0, indices[0])] = valor;
				break;
			
			case 2:
				dados[indice(0, 0, indices[0], indices[1])] = valor;
				break;
			
			case 3:
				dados[indice(0, indices[0], indices[1], indices[2])] = valor;
				break;
			
			case 4:
				dados[indice(indices[0], indices[1], indices[2], indices[3])] = valor;
				break;
		
			default:
				throw new IndexOutOfBoundsException(
					STR."\nQuantidade de índices (\{indices.length}) inválida"
				);
		}
	}

	/**
	 * Preenche todo o conteúdo do tensor com um valor constante.
	 * @param valor valor desejado.
	 * @return instância local alterada.
	 */
	public Tensor4D preencher(double valor) {
		for (int i = 0; i < dados.length; i++) {
			dados[i] = valor;
		}

		return this;
	}

	/**
	 * Preenche o conteúdo desejado do tensor com um valor constante.
	 * @param dim1 índice da primeira dimensão.
	 * @param valor valor desejado.
	 * @return instância local alterada.
	 */
	public Tensor4D preencher3D(int dim1, double valor) {
		verificarIndiceD1(dim1);

		int inicio = indice(dim1, 0, 0, 0);
		int fim = inicio + (dim2() * dim3() * dim4());

		for (int i = inicio; i < fim; i++) {
			dados[i] = valor;
		}

		return this;
	}

	/**
	 * Preenche o conteúdo desejado do tensor com um valor constante.
	 * @param dim1 índice da primeira dimensão.
	 * @param dim2 índice da segunda dimensão.
	 * @param valor valor desejado.
	 * @return instância local alterada.
	 */
	public Tensor4D preencher2D(int dim1, int dim2, double valor) {
		verificarIndiceD1(dim1);
		verificarIndiceD2(dim2);

		int inicio = indice(dim1, dim2, 0, 0);
		int fim = inicio + (dim3() * dim4());

		for (int i = inicio; i < fim; i++) {
			dados[i] = valor;
		}

		return this;
	}

	/**
	 * Preenche o conteúdo desejado do tensor com um valor constante.
	 * @param dim1 índice da primeira dimensão.
	 * @param dim2 índice da segunda dimensão.
	 * @param dim3 índice da terceira dimensão.
	 * @param valor valor desejado.
	 * @return instância local alterada.
	 */
	public Tensor4D preencher1D(int dim1, int dim2, int dim3, double valor) {
		verificarIndiceD1(dim1);
		verificarIndiceD2(dim2);
		verificarIndiceD3(dim3);

		int inicio = indice(dim1, dim2, dim3, 0);
		int fim = inicio + dim4();

		for (int i = inicio; i < fim; i++) {
			dados[i] = valor;
		}

		return this;
	}

	/**
	 * Preenche o conteúdo do tensor usando um contador iniciado com
	 * valor 1 que é alterado a cada elemento.
	 * @param cres contador crescente (1, 2, 3, ...), caso falso o
	 * contador é decrescente (-1, -2, -3, ...).
	 * @return instância local alterada.
	 */
	public Tensor4D preencherContador(boolean cres) {
		int tam = tamanho();

		if (cres) {
			for (int i = 0; i < tam; i++) {
				dados[i] = i + 1;
			}

		} else {
			for (int i = 0; i < tam; i++) {
				dados[i] = tam - i - 1;
			}
		}

		return this;
	}

	/**
	 * Zera todo o conteúdo o tensor.
	 * @return instância local alterada.
	 */
	public Tensor4D zerar() {
		for (int i = 0; i < dados.length; i++) {
			dados[i] = 0.0d;
		}

		return this;
	}

	/**
	 * Copia todo o conteúdo do tensor na instância local.
	 * @param tensor tensor desejado.
	 * @return instância local alterada.
	 */
	public Tensor4D copiar(Tensor4D tensor) {
		if (!comparar4D(tensor)) {
			throw new IllegalArgumentException(
				"\nDimensões " + shapeStr() + " incompatíveis com as do" +
				" tensor recebido " + tensor.shapeStr()
			);
		}

		System.arraycopy(tensor.dados, 0, this.dados, 0, tamanho());

		return this;
	}

	/**
	 * Copia o conteúdo do tensor na instância local de acordo a dimensão fornecida.
	 * @param tensor tensor desejado.
	 * @param dim1 índice da primeira dimensão desejada.
	 * @return instância local alterada.
	 */
	public Tensor4D copiar(Tensor4D tensor, int dim1) {
		if (!comparar3D(tensor)) {
			throw new IllegalArgumentException(
				"\nIncompatibilidade entre as três últimas dimensões do tensor " + shapeStr() +
				" com o tensor fornecido " + tensor.shapeStr()
			);
		}

		int inicio = indice(dim1, 0, 0, 0);
		System.arraycopy(tensor.dados, 0, this.dados, inicio, (dim2() * dim3() * dim4()));

		return this;
	}

	/**
	 * Copia o conteúdo do tensor na instância local de acordo as dimensões
	 * fornecidas.
	 * @param tensor tensor desejado.
	 * @param dim1 índice da primeira dimensão desejada.
	 * @param dim2 índice da segunda dimensão desejada.
	 * @return instância local alterada.
	 */
	public Tensor4D copiar(Tensor4D tensor, int dim1, int dim2) {
		if (!comparar2D(tensor)) {
			throw new IllegalArgumentException(
				"\nIncompatibilidade entre as duas últimas dimensões do tensor " + shapeStr() +
				" com o tensor fornecido " + tensor.shapeStr()
			);
		}

		int inicio = indice(dim1, dim2, 0, 0);
		System.arraycopy(tensor.dados, 0, this.dados, inicio, (dim3() * dim4()));

		return this;
	}

	/**
	 * Copia o conteúdo do tensor na instância local de acordo as dimensões
	 * fornecidas.
	 * @param tensor tensor desejado.
	 * @param dim1 índice da primeira dimensão desejada.
	 * @param dim2 índice da segunda dimensão desejada.
	 * @param dim3 índice da terceira dimensão desejada.
	 * @return instância local alterada.
	 */
	public Tensor4D copiar(Tensor4D tensor, int dim1, int dim2, int dim3) {
		if (!comparar1D(tensor)) {
			throw new IllegalArgumentException(
				"\nIncompatibilidade entre a última dimensão do tensor " + shapeStr() +
				" com o tensor fornecido " + tensor.shapeStr()
			);
		}

		int inicio = indice(dim1, dim2, dim3, 0);
		System.arraycopy(tensor.dados, 0, this.dados, inicio, dim4());

		return this;
	}

	/**
	 * Copia todo o conteúdo do array na instância local.
	 * @param arr array desejado.
	 * @return instância local alterada.
	 */
	public Tensor4D copiar(double[][][][] arr) {
		if ((dim1() != arr.length) ||
			(dim2() != arr[0].length) ||
			(dim3() != arr[0][0].length) ||
			(dim4() != arr[0][0][0].length)) {
			throw new IllegalArgumentException(
				"\nDimensões do tensor " + shapeStr() +
				" incompatíveis com as do array recebido ("
				+ arr.length + ", " + arr[0].length + ", " + arr[0][0].length + ", " + arr[0][0][0].length
				+ ")."
			);
		}

		int cont = 0;
		for (int i = 0; i < dim1(); i++) {
			for (int j = 0; j < dim2(); j++) {
				for (int k = 0; k < dim3(); k++) {
					for (int l = 0; l < dim4(); l++) {
						this.dados[cont++] = arr[i][j][k][l];
					}
				}
			}
		}

		return this;
	}

	/**
	 * Copia todo o conteúdo do array na instância local.
	 * @param arr  array desejado.
	 * @param dim1 índice da primeira dimensão.
	 * @return instância local alterada.
	 */
	public Tensor4D copiar(double[][][] arr, int dim1) {
		if ((dim2() != arr.length) ||
			(dim3() != arr[0].length) ||
			(dim4() != arr[0][0].length)) {
			throw new IllegalArgumentException(
				"\nDimensões do tensor " + shapeStr() +
				" incompatíveis com as do array recebido ("
				+ arr.length + ", " + arr[0].length + ", " + arr[0][0].length + ")."
			);
		}

		verificarIndiceD1(dim1);

		int inicio = indice(dim1, 0, 0, 0);
		for (int j = 0; j < dim2(); j++) {
			for (int k = 0; k < dim3(); k++) {
				for (int l = 0; l < dim4(); l++) {
					this.dados[inicio++] = arr[j][k][l];
				}
			}
		}

		return this;
	}

	/**
	 * Copia o conteúdo do array na instância local.
	 * @param arr array desejado.
	 * @param dim1 índice da primeira dimensão.
	 * @param dim2 índice da segunda dimensão.
	 * @return instância local alterada.
	 */
	public Tensor4D copiar(double[][] arr, int dim1, int dim2) {
		if ((dim3() != arr.length) || (dim4() != arr[0].length)) {
			throw new IllegalArgumentException(
				"\nDimensões do tensor " + shapeStr() +
				" incompatíveis com as do array recebido ("
				+ arr.length + ", " + arr[0].length + ")."
			);
		}

		verificarIndiceD1(dim1);
		verificarIndiceD2(dim2);

		for (int i = 0; i < dim3(); i++) {
			int inicio = indice(dim1, dim2, i, 0);
			System.arraycopy(arr[i], 0, dados, inicio, dim4());
		}

		return this;
	}

	/**
	 * Copia o conteúdo do array na instância local.
	 * @param arr array desejado.
	 * @param dim1 índice da primeira dimensão.
	 * @param dim2 índice da segunda dimensão.
	 * @param dim3 índice da terceira dimensão.
	 * @return instância local alterada.
	 */
	public Tensor4D copiar(double[] arr, int dim1, int dim2, int dim3) {
		if (arr.length != dim4()) {
			throw new IllegalArgumentException(
				"\nTamanho do array (" + arr.length +
				") íncompatível com a capacidade do tensor (" + dim4() + ")."
			);
		}

		verificarIndiceD1(dim1);
		verificarIndiceD2(dim2);
		verificarIndiceD3(dim3);

		int inicio = indice(dim1, dim2, dim3, 0);
		System.arraycopy(arr, 0, dados, inicio, dim4());

		return this;
	}

	/**
	 * Copia apenas os dados contidos no array, sem levar em consideração
	 * as dimensões do tensor.
	 * <p>
	 * Ainda é necessário que a quantidade de elementos do array seja igual
	 * a quantidade de elementos do tensor.
	 * </p>
	 * @param elementos array de elementos desejado.
	 * @return instância local alterada.
	 */
	public Tensor4D copiarElementos(double[] elementos) {
		if (elementos == null) {
			throw new IllegalArgumentException(
				"\nArray de elementos não pode ser nulo."
			);
		}

		if (elementos.length != tamanho()) {
			throw new IllegalArgumentException(
				"\nTamanho do array fornecido (" + elementos.length + ") inconpatível" +
				"com os elementos do tensor (" + tamanho() + ")."
			);
		}

		System.arraycopy(elementos, 0, dados, 0, tamanho());

		return this;
	}

	/**
	 * Verifica se as dimensões selecionada são iguais as da instância local.
	 * @param tensor tensor base.
	 * @param dim1 comparar primeira dimensão.
	 * @param dim2 comparar segunda dimensão.
	 * @param dim3 comparar terceira dimensão.
	 * @param dim4 comparar quarta dimensão.
	 * @return resultado da verifcação.
	 */
	private boolean compararDimensoes(Tensor4D tensor, boolean dim1, boolean dim2, boolean dim3, boolean dim4) {
		if (dim4 && (tensor.dim4() != this.dim4())) return false;
		if (dim3 && (tensor.dim3() != this.dim3())) return false;
		if (dim2 && (tensor.dim2() != this.dim2())) return false;
		if (dim1 && (tensor.dim1() != this.dim1())) return false;

		return true;
	}

	/**
	 * Verifica todas as quatro dimensões do tensor local com os
	 * valores de dimensões do tensor recebido.
	 * @param tensor tensor alvo.
	 * @return resultado da verificação.
	 */
	public boolean comparar4D(Tensor4D tensor) {
		return compararDimensoes(tensor, true, true, true, true);
	}

	/**
	 * Verifica as três últimas dimensões do tensor local com os
	 * valores de dimensões do tensor recebido.
	 * @param tensor tensor alvo.
	 * @return resultado da verificação.
	 */
	public boolean comparar3D(Tensor4D tensor) {
		return compararDimensoes(tensor, false, true, true, true);
	}

	/**
	 * Verifica as duas últimas dimensões do tensor local com os
	 * valores de dimensões do tensor recebido.
	 * @param tensor tensor alvo.
	 * @return resultado da verificação.
	 */
	public boolean comparar2D(Tensor4D tensor) {
		return compararDimensoes(tensor, false, false, true, true);
	}

	/**
	 * Verifica a última dimensõe do tensor local com os
	 * valores de dimensões do tensor recebido.
	 * @param tensor tensor alvo.
	 * @return resultado da verificação.
	 */
	public boolean comparar1D(Tensor4D tensor) {
		return compararDimensoes(tensor, false, false, false, true);
	}

	/**
	 * Compara todo o conteúdo da instância local, isso inclui as {@code dimensões}
	 * de cada tensor e seus {@code elementos individuais}.
	 * @param tensor tensor base.
	 * @return {@code true} caso sejam iguais, {@code false} caso contrário.
	 */
	public boolean comparar(Tensor4D tensor) {
		if (!comparar4D(tensor)) return false;

		for (int i = 0; i < dados.length; i++) {
			if (dados[i] != tensor.dados[i]) return false;
		}

		return true;
	}

	/**
	 * Retorna o valor contido no tensor, caso ele possua apenas um elemento.
	 * @return valor contido no tensor.
	 */
	public double item() {
		if (tamanho() > 1) {
			throw new IllegalArgumentException(
				"\nO tensor deve conter apenas um elemento."
			);
		}

		return dados[0];
	}

	/**
	 * Aplica a função recebida em todos os elementos do tensor.
	 * <p>
	 *		Exemplo:
	 * </p>
	 * <pre>
	 * tensor.map(x -> Math.random());
	 * </pre>
	 * Onde {@code x} representa cada elemento dentro do tensor local.
	 * @param fun função desejada.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor4D map(DoubleUnaryOperator fun) {
		if (fun == null) {
			throw new IllegalArgumentException(
				"\nFunção recebida é nula."
			);
		}

		Tensor4D t = new Tensor4D(shape());

		for (int i = 0; i < t.tamanho(); i++) {
			t.dados[i] = fun.applyAsDouble(dados[i]);
		}

		return t;
	}

	/**
	 * Reduz os elementos do tensor para um, aplicando a função de recebida.
	 * <p>
	 * Exemplo:
	 * </p>
	 * <pre>
	 *tensor = {1, 2, 3, 4, 5};
	 *res = tensor.reduce(0, (x, y) -> x+y);//tensor = {15}
	 * </pre>
	 * @param in valor inicial.
	 * @param fun função desejada.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor4D reduce(double in, DoubleBinaryOperator fun) {
		if (fun == null) {
			throw new IllegalArgumentException(
				"\nFunção de redução não pode ser nula."
			);
		}

		double res = in;
		for (double val : dados) {
			res = fun.applyAsDouble(res, val);
		}

		return new Tensor4D(new double[]{ res });
	}

	/**
	 * Aplica a função recebida em todos os elementos do tensor usando
	 * como entrada os valores do tensor recebido.
	 * <p>
	 * Exemplo:
	 * </p>
	 * <pre>
	 * tensor.aplicar(x -> Math.random());
	 * </pre>
	 * Onde {@code x} representa cada elemento dentro do tensor fornecido.
	 * @param tensor tensor base.
	 * @param fun função para aplicar no tensor base.
	 * @return instância local alterada.
	 */
	public Tensor4D aplicar(Tensor4D tensor, DoubleUnaryOperator fun) {
		if (tensor == null) {
			throw new IllegalArgumentException(
				"\nTensor fornecido é nulo."
			);
		}
		if (!comparar4D(tensor)) {
			throw new IllegalArgumentException(
				"\nAs dimensões do tensor fornecido " + tensor.shapeStr() +
				" e as da instância local " + shapeStr() + " devem ser iguais."
			);
		}
		if (fun == null) {
			throw new IllegalArgumentException(
				"\nFunção recebida é nula."
			);
		}

		for (int i = 0; i < dados.length; i++) {
			dados[i] = fun.applyAsDouble(tensor.dados[i]);
		}

		return this;
	}

	/**
	 * Aplica a função recebida em todos os elementos do tensor.
	 * <p>
	 * Exemplo:
	 * </p>
	 * <pre>
	 * tensor.aplicar(x -> Math.random());
	 * </pre>
	 * Onde {@code x} representa cada elemento dentro do tensor.
	 * 
	 * @param fun função desejada.
	 * @return instância local alterada.
	 */
	public Tensor4D aplicar(DoubleUnaryOperator fun) {
		if (fun == null) {
			throw new IllegalArgumentException(
				"\nFunção recebida é nula."
			);
		}

		for (int i = 0; i < dados.length; i++) {
			dados[i] = fun.applyAsDouble(dados[i]);
		}

		return this;
	}

	/**
	 * Aplica a função recebida em todos os elementos da primeira dimensão
	 * do tensor.
	 * <p>
	 * Exemplo:
	 * </p>
	 * <pre>
	 * tensor.aplicar(0, x -> Math.random());
	 * </pre>
	 * Onde {@code x} representa cada elemento dentro do tensor.
	 * @param dim1 índice da primeira dimensão.
	 * @param func função desejada.
	 * @return instância local alterada.
	 */
	public Tensor4D aplicar(int dim1, DoubleUnaryOperator func) {
		verificarIndiceD1(dim1);

		if (func == null) {
			throw new IllegalArgumentException(
				"\nFunção recebida é nula."
			);
		}

		int inicio = indice(dim1, 0, 0, 0);
		int fim = inicio + (dim2() * dim3() * dim4());

		for (int i = inicio; i < fim; i++) {
			dados[i] = func.applyAsDouble(dados[i]);
		}

		return this;
	}

	/**
	 * Aplica a função recebida em todos os elementos da segunda dimensão
	 * do tensor.
	 * <p>
	 * Exemplo:
	 * </p>
	 * <pre>
	 * tensor.aplicar(0, 0, x -> Math.random());
	 * </pre>
	 * Onde {@code x} representa cada elemento dentro do tensor.
	 * @param dim1 índice da primeira dimensão.
	 * @param dim2 índice da segunda dimensão.
	 * @param fun função desejada.
	 * @return instância local alterada.
	 */
	public Tensor4D aplicar(int dim1, int dim2, DoubleUnaryOperator fun) {
		verificarIndiceD1(dim1);
		verificarIndiceD2(dim2);

		if (fun == null) {
			throw new IllegalArgumentException(
				"\nFunção recebida é nula."
			);
		}

		int inicio = indice(dim1, dim2, 0, 0);
		int fim = inicio + (dim3() * dim4());

		for (int i = inicio; i < fim; i++) {
			dados[i] = fun.applyAsDouble(dados[i]);
		}

		return this;
	}

	/**
	 * Aplica a função recebida em todos os elementos da terceira dimensão
	 * do tensor.
	 * <p>
	 * Exemplo:
	 * </p>
	 * <pre>
	 * tensor.aplicar(0, 0, 0, x -> Math.random());
	 * </pre>
	 * Onde {@code x} representa cada elemento dentro do tensor.
	 * @param dim1 índice da primeira dimensão.
	 * @param dim2 índice da segunda dimensão.
	 * @param dim3 índice da terceira dimensão.
	 * @param fun função desejada.
	 * @return instância local alterada.
	 */
	public Tensor4D aplicar(int dim1, int dim2, int dim3, DoubleUnaryOperator fun) {
		verificarIndiceD1(dim1);
		verificarIndiceD2(dim2);
		verificarIndiceD3(dim3);

		if (fun == null) {
			throw new IllegalArgumentException(
				"\nFunção recebida é nula."
			);
		}

		int inicio = indice(dim1, dim2, dim3, 0);
		int fim = inicio + dim4();
		for (int i = inicio; i < fim; i++) {
			dados[i] = fun.applyAsDouble(dados[i]);
		}

		return this;
	}

	/**
	 * Retorna a {@code soma} de todos os elementos do tensor.
	 * @return soma total.
	 */
	public double soma() {
		double soma = 0;

		for (double val : dados) {
			soma += val;
		}

		return soma;
	}

	/**
	 * Retorna a {@code média} de todos os elementos do tensor.
	 * @return média entre os valores do tensor.
	 */
	public double media() {
		return soma() / tamanho();
	}

	/**
	 * Retorna o {@code valor máximo} contido no tensor.
	 * @return valor máximo.
	 */
	public double maximo() {
		double max = dados[0];
		int tam = tamanho();

		for (int i = 1; i < tam; i++) {
			if (dados[i] > max) max = dados[i];
		}

		return max;
	}

	/**
	 * Retorna o {@code valor minimo} contido no tensor.
	 * @return valor minimo.
	 */
	public double minimo() {
		double min = dados[0];
		int tam = tamanho();

		for (int i = 1; i < tam; i++) {
			if (dados[i] < min) min = dados[i];
		}

		return min;
	}

	/**
	 * Retorna o valor do {@code devio padrão} dos dos elementos
	 * do tensor.
	 * @return desvio padrão.
	 */
	public double desvp() {
		double media = media();
		double soma = 0.0d;

		for (int i = 0; i < dados.length; i++) {
			soma += Math.pow(dados[i] - media, 2);
		}

		return Math.sqrt(soma / tamanho());
	}

	/**
	 * Aplica a função argmax na última dimensão do tensor.
	 * <p>
	 * 		Exemplo:
	 * </p>
	 * <pre>
	 *tensor = {
	 * {1, 2, 3},
	 * {3, 2, 1},
	 *}
	 * 
	 *argmax = {
	 * {0, 0, 1},
	 * {1, 0, 0},
	 *}
	 * </pre>
	 * @return instância local alterada.
	 */
	public Tensor4D argmax() {
		for (int i = 0; i < dim1(); i++) {
			for (int j = 0; j < dim2(); j++) {
				for (int k = 0; k < dim3(); k++) {
					double maxVal = get(i, j, k, 0);
					int maxId = 0;

					for (int l = 1; l < dim4(); l++) {
						if (get(i, j, k, l) > maxVal) {
							maxVal = get(i, j, k, l);
							maxId = l;
						}
					}

					for (int l = 0; l < dim4(); l++) {
						set(
							(l == maxId ? 1 : 0),
							i, j, k, l
						);
					}

				}
			}
		}

		return this;
	}

	/**
	 * Retorna a soma de todos os elementos das últimas três dimensões
	 * do tensor de acordo com o índice especificado.
	 * @param dim1 índice da primeira dimensão do tensor.
	 * @return soma total.
	 */
	public double soma3D(int dim1) {
		verificarIndiceD1(dim1);

		int inicio = indice(dim1, 0, 0, 0);
		int fim = inicio + (dim2() * dim3() * dim4());
		double soma = 0;

		for (int i = inicio; i < fim; i++) {
			soma += dados[i];
		}

		return soma;
	}

	/**
	 * Retorna a soma de todos os elementos das últimas duas dimensões
	 * do tensor de acordo com os índices especificados.
	 * @param dim1 índice da primeira dimensão do tensor.
	 * @param dim2 índice da segunda dimensão do tensor.
	 * @return soma total.
	 */
	public double soma2D(int dim1, int dim2) {
		verificarIndiceD1(dim1);
		verificarIndiceD2(dim2);

		int inicio = indice(dim1, dim2, 0, 0);
		int fim = inicio + (dim3() * dim4());
		double soma = 0;

		for (int i = inicio; i < fim; i++) {
			soma += dados[i];
		}

		return soma;
	}

	/**
	 * Retorna a soma de todos os elementos da última dimensão
	 * do tensor de acordo com os índices especificados.
	 * @param dim1 índice da primeira dimensão do tensor.
	 * @param dim2 índice da segunda dimensão do tensor.
	 * @param dim3 índice da terceira dimensão do tensor.
	 * @return soma total.
	 */
	public double soma1D(int dim1, int dim2, int dim3) {
		verificarIndiceD1(dim1);
		verificarIndiceD2(dim2);
		verificarIndiceD3(dim3);

		int inicio = indice(dim1, dim2, dim3, 0);
		int fim = inicio + dim4();
		double soma = 0;

		for (int i = inicio; i < fim; i++) {
			soma += dados[i];
		}

		return soma;
	}

	/**
	 * Copia o conteúdo de uma linha do tensor e repete ela na quantidade fornecida.
	 * <p>
	 * Exemplo:
	 * </p>
	 * <pre>
	 *tensor = [
	 * [
	 *  [[1, 2, 3]]  
	 * ]  
	 *]
	 *
	 *bloco = bloco2D(0, 0, 0, 3);
	 *
	 *tensorBloco = [
	 * [
	 *   [[1, 2, 3],
	 *    [1, 2, 3],
	 *    [1, 2, 3]]
	 * ]  
	 *]
	 * </pre>
	 * @param dim1 índice da primeira dimensão.
	 * @param dim2 índice da segunda dimensão.
	 * @param dim3 índice da terceira dimensão.
	 * @param n quantidade de repetições.
	 * @return {@code novo tensor} com o resultado.
	 */
	public Tensor4D bloco(int dim1, int dim2, int dim3, int n) {
		verificarIndiceD1(dim1);
		verificarIndiceD2(dim2);
		verificarIndiceD3(dim3);

		Tensor4D bloco = new Tensor4D(1, 1, n, dim4());
		double[] arr = array1D(dim1, dim2, dim3);

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < dim4(); j++) {
				bloco.set(arr[j], 0, 0, i, j);
			}
		}

		return bloco;
	}

	/**
	 * Transoforma o conteúdo das últimas duas dimensões do tensor em uma matriz
	 * identidade.
	 * <p>
	 * Na matriz identidade todos os valores são zerados e os valores da diagonal
	 * principal são editados para 1.
	 * </p>
	 * @param dim1 índice da primeira dimensão.
	 * @param dim2 índice da segunda dimensão.
	 * @return instância local alterada.
	 */
	public Tensor4D identidade(int dim1, int dim2) {
		verificarIndiceD1(dim1);
		verificarIndiceD2(dim2);

		for (int i = 0; i < dim3(); i++) {
			for (int j = 0; j < dim4(); j++) {
				set((i == j ? 1 : 0), dim1, dim2, i, j);
			}
		}

		return this;
	}

	/**
	 * Adiciona todo o conteúdo {@code elemento a elemento} do tensor recebido,
	 * seguindo a expressão:
	 * <pre>
	 *  this += tensor
	 * </pre>
	 * @param tensor tensor com conteúdo.
	 * @return instância local alterada.
	 */
	public Tensor4D add(Tensor4D tensor) {
		if (!comparar4D(tensor)) {
			throw new IllegalArgumentException(
				"\nDimensões " + shapeStr() + " incompatíveis com as do" +
				" tensor recebido " + tensor.shapeStr()
			);
		}

		int tam = tamanho();
		for (int i = 0; i < tam; i++) {
			dados[i] += tensor.dados[i];
		}

		return this;
	}

	/**
	 * Adiciona o valor ao conteúdo do tensor de acordo com os índices fornecidos;
	 * @param d1 índice da primeira dimensão.
	 * @param d2 índice da segunda dimensão.
	 * @param d3 índice da terceira dimensão.
	 * @param d4 índice da quarta dimensão.
	 * @param valor valor desejado.
	 * @return instância local alterada.
	 */
	public Tensor4D add(int d1, int d2, int d3, int d4, double valor) {
		dados[indice(d1, d2, d3, d4)] += valor;
		return this;
	}

	/**
	 * Adiciona o valor fornecido ao conteúdo das duas últimas dimensões do tensor.
	 * @param dim1 índice da primeira dimensão.
	 * @param dim2 índice da segunda dimensão.
	 * @param valor valor desejado.
	 * @return instância local alterada.
	 */
	public Tensor4D add2D(int dim1, int dim2, double valor) {
		verificarIndiceD1(dim1);
		verificarIndiceD2(dim2);

		int inicio = indice(dim1, dim2, 0, 0);
		int fim = inicio + (dim3() * dim4());

		for (int i = inicio; i < fim; i++) {
			dados[i] += valor;
		}

		return this;
	}

	/**
	 * Subtrai todo o conteúdo {@code elemento a elemento} do tensor recebido,
	 * seguindo a expressão:
	 * <pre>
	 * this -= tensor
	 * </pre>
	 * @param tensor tensor com conteúdo.
	 * @return instância local alterada.
	 */
	public Tensor4D sub(Tensor4D tensor) {
		if (!comparar4D(tensor)) {
			throw new IllegalArgumentException(
				"\nDimensões " + shapeStr() + " incompatíveis com as do" +
				" tensor recebido " + tensor.shapeStr()
			);
		}

		int tam = tamanho();
		for (int i = 0; i < tam; i++) {
			dados[i] -= tensor.dados[i];
		}

		return this;
	}

	/**
	 * Subtrai o valor ao conteúdo do tensor de acordo com os índices fornecidos;
	 * @param d1 índice da primeira dimensão.
	 * @param d2 índice da segunda dimensão.
	 * @param d3 índice da terceira dimensão.
	 * @param d4 índice da quarta dimensão.
	 * @param valor valor desejado.
	 * @return instância local alterada.
	 */
	public Tensor4D sub(int d1, int d2, int d3, int d4, double valor) {
		dados[indice(d1, d2, d3, d4)] -= valor;
		return this;
	}

	/**
	 * Subtrai o valor fornecido ao conteúdo das duas últimas dimensões do tensor.
	 * @param dim1 índice da primeira dimensão.
	 * @param dim2 índice da segunda dimensão.
	 * @param valor valor desejado.
	 * @return instância local alterada.
	 */
	public Tensor4D sub2D(int dim1, int dim2, double valor) {
		verificarIndiceD1(dim1);
		verificarIndiceD2(dim2);

		int inicio = indice(dim1, dim2, 0, 0);
		int fim = inicio + (dim3() * dim4());

		for (int i = inicio; i < fim; i++) {
			dados[i] -= valor;
		}

		return this;
	}

	/**
	 * Multiplica todo o conteúdo {@code elemento a elemento} do tensor recebido,
	 * seguindo
	 * a expressão:
	 * <pre>
	 * this *= tensor
	 * </pre>
	 * @param tensor tensor com conteúdo.
	 * @return instância local alterada.
	 */
	public Tensor4D mult(Tensor4D tensor) {
		if (!comparar4D(tensor)) {
			throw new IllegalArgumentException(
				"\nDimensões " + shapeStr() + " incompatíveis com as do" +
				" tensor recebido " + tensor.shapeStr()
			);
		}

		int tam = tamanho();
		for (int i = 0; i < tam; i++) {
			dados[i] *= tensor.dados[i];
		}

		return this;
	}

	/**
	 * Multiplica o valor ao conteúdo do tensor de acordo com os índices fornecidos;
	 * @param d1 índice da primeira dimensão.
	 * @param d2 índice da segunda dimensão.
	 * @param d3 índice da terceira dimensão.
	 * @param d4 índice da quarta dimensão.
	 * @param valor valor desejado.
	 * @return instância local alterada.
	 */
	public Tensor4D mult(int d1, int d2, int d3, int d4, double valor) {
		dados[indice(d1, d2, d3, d4)] *= valor;
		return this;
	}

	/**
	 * Multiplica o valor fornecido ao conteúdo das duas últimas dimensões do
	 * tensor.
	 * @param dim1 índice da primeira dimensão.
	 * @param dim2 índice da segunda dimensão.
	 * @param valor valor desejado.
	 * @return instância local alterada.
	 */
	public Tensor4D mult2D(int dim1, int dim2, double valor) {
		verificarIndiceD1(dim1);
		verificarIndiceD2(dim2);

		int inicio = indice(dim1, dim2, 0, 0);
		int fim = inicio + (dim3() * dim4());

		for (int i = inicio; i < fim; i++) {
			dados[i] *= valor;
		}

		return this;
	}

	/**
	 * Divide todo o conteúdo {@code elemento a elemento} do tensor recebido,
	 * seguindo a expressão:
	 * <pre>
	 * this /= tensor
	 * </pre>
	 * @param tensor tensor com conteúdo.
	 * @return instância local alterada.
	 */
	public Tensor4D div(Tensor4D tensor) {
		if (!comparar4D(tensor)) {
			throw new IllegalArgumentException(
				"\nDimensões " + shapeStr() + " incompatíveis com as do" +
				" tensor recebido " + tensor.shapeStr()
			);
		}

		int tam = tamanho();
		for (int i = 0; i < tam; i++) {
			dados[i] /= tensor.dados[i];
		}

		return this;
	}

	/**
	 * Divide o valor ao conteúdo do tensor de acordo com os índices fornecidos,
	 * como
	 * no exemplo:
	 * <pre>
	 * tensor[i][j][k][l] /= valor;
	 * </pre>
	 * @param d1 índice da primeira dimensão.
	 * @param d2 índice da segunda dimensão.
	 * @param d3 índice da terceira dimensão.
	 * @param d4 índice da quarta dimensão.
	 * @param valor valor desejado.
	 * @return instância local alterada.
	 */
	public Tensor4D div(int d1, int d2, int d3, int d4, double valor) {
		dados[indice(d1, d2, d3, d4)] /= valor;
		return this;
	}

	/**
	 * Divide o valor fornecido ao conteúdo das duas últimas dimensões do tensor.
	 * @param dim1 índice da primeira dimensão.
	 * @param dim2 índice da segunda dimensão.
	 * @param valor valor desejado.
	 * @return instância local alterada.
	 */
	public Tensor4D div2D(int dim1, int dim2, double valor) {
		verificarIndiceD1(dim1);
		verificarIndiceD2(dim2);

		int inicio = indice(dim1, dim2, 0, 0);
		int fim = inicio + (dim3() * dim4());

		for (int i = inicio; i < fim; i++) {
			dados[i] /= valor;
		}

		return this;
	}

	/**
	 * Retorna o conteúdo do tensor no formato de array
	 * @return conteúdo do tensor.
	 */
	public double[] paraArray() {
		return dados.clone();
	}

	/**
	 * Retorna um array de uma dimensão de acordo com os índices
	 * especificados.
	 * @param dim1 índice da primeira dimensão.
	 * @param dim2 índice da segunda dimensão.
	 * @param dim3 índice da terceira dimensão.
	 * @return array contendo os elementos.
	 */
	public double[] array1D(int dim1, int dim2, int dim3) {
		verificarIndiceD1(dim1);
		verificarIndiceD2(dim2);
		verificarIndiceD3(dim3);

		double[] res = new double[dim4()];

		int inicio = indice(dim1, dim2, dim3, 0);
		System.arraycopy(this.dados, inicio, res, 0, dim4());

		return res;
	}

	/**
	 * Retorna um array de duas dimensões de acordo com os índices
	 * especificados.
	 * @param dim1 índice da primeira dimensão.
	 * @param dim2 índice da segunda dimensão.
	 * @return array contendo os elementos.
	 */
	public double[][] array2D(int dim1, int dim2) {
		verificarIndiceD1(dim1);
		verificarIndiceD2(dim2);

		double[][] res = new double[dim3()][dim4()];

		int inicio;
		for (int i = 0; i < dim3(); i++) {
			inicio = indice(dim1, dim2, i, 0);
			System.arraycopy(this.dados, inicio, res[i], 0, dim4());
		}

		return res;
	}

	/**
	 * Retorna um array de três dimensões de acordo com o índice
	 * especificado.
	 * @param dim1 índice da primeira dimensão.
	 * @return array contendo os elementos.
	 */
	public double[][][] array3D(int dim1) {
		verificarIndiceD1(dim1);

		double[][][] res = new double[dim2()][dim3()][dim4()];

		int inicio;
		for (int i = 0; i < dim2(); i++) {
			for (int j = 0; j < dim3(); j++) {
				inicio = indice(dim1, i, j, 0);
				System.arraycopy(this.dados, inicio, res[i][j], 0, dim4());
			}
		}

		return res;
	}

	/**
	 * Retorna um array de quatro dimensões contendo o todo o conteúdo
	 * do tensor.
	 * @return array contendo os elementos.
	 */
	public double[][][][] array4D() {
		double[][][][] res = new double[dim1()][dim2()][dim3()][dim4()];

		for (int i = 0; i < dim1(); i++) {
			for (int j = 0; j < dim2(); j++) {
				for (int k = 0; k < dim3(); k++) {
					for (int l = 0; l < dim4(); l++) {
						res[i][j][k][l] = get(i, j, k, l);
					}
				}
			}
		}

		return res;
	}

	/**
	 * Retorna um novo tensor contendo o conteúdo da dimensão especificada.
	 * @param dim1 índice da primeira dimensão desejada.
	 * @return tensor contendo os sub dados.
	 */
	public Tensor4D subTensor3D(int dim1) {
		verificarIndiceD1(dim1);

		Tensor4D tensor = new Tensor4D(dim2(), dim3(), dim4());

		int inicio = indice(dim1, 0, 0, 0);
		System.arraycopy(this.dados, inicio, tensor.dados, 0, (dim2() * dim3() * dim4()));

		return tensor;
	}

	/**
	 * Retorna um novo tensor contendo o conteúdo das dimensões especificadas.
	 * @param dim1 índice da primeira dimensão desejada.
	 * @param dim2 índice da segunda dimensão desejada.
	 * @return tensor contendo os sub dados.
	 */
	public Tensor4D subTensor2D(int dim1, int dim2) {
		verificarIndiceD1(dim1);
		verificarIndiceD2(dim2);

		Tensor4D tensor = new Tensor4D(dim3(), dim4());

		int inicio = indice(dim1, dim2, 0, 0);
		System.arraycopy(this.dados, inicio, tensor.dados, 0, (dim3() * dim4()));

		return tensor;
	}

	/**
	 * Retorna um novo tensor contendo o conteúdo das dimensões especificadas.
	 * @param dim1 índice da primeira dimensão desejada.
	 * @param dim2 índice da segunda dimensão desejada.
	 * @param dim3 índice da terceira dimensão desejada.
	 * @return tensor contendo os sub dados.
	 */
	public Tensor4D subTensor1D(int dim1, int dim2, int dim3) {
		verificarIndiceD1(dim1);
		verificarIndiceD2(dim2);
		verificarIndiceD3(dim3);

		Tensor4D tensor = new Tensor4D(dim4());

		int inicio = indice(dim1, dim2, dim3, 0);
		System.arraycopy(this.dados, inicio, tensor.dados, 0, dim4());

		return tensor;
	}

	/**
	 * Normaliza os valores do tensor dentro do intervalo especificado.
	 * @param min valor mínimo do intervalo.
	 * @param max valor máximo do intervalo.
	 * @return instância local alterada.
	 */
	public Tensor4D normalizar(double min, double max) {
		double valMin = minimo();
		double valMax = maximo();

		double intOriginal = valMax - valMin;
		double intNovo = max - min;

		for (int i = 0; i < dados.length; i++) {
			dados[i] = ((dados[i] - valMin) / intOriginal) * intNovo + min;
		}

		return this;
	}

	/**
	 * Aplica a função de ativação {@code ReLU} em todos os
	 * elementos do tensor.
	 * @return instância local alterada.
	 */
	public Tensor4D relu() {
		return aplicar(x -> x > 0 ? x : 0);
	}

	/**
	 * Aplica a função de ativação {@code Sigmoid} em todos os
	 * elementos do tensor.
	 * @return instância local alterada.
	 */
	public Tensor4D sigmoid() {
		return aplicar(x -> 1 / (1 + Math.exp(-x)));
	}

	/**
	 * Aplica a função de ativação {@code TanH} (Tangente Hiperbólica)
	 * em todos os elementos do tensor.
	 * @return instância local alterada.
	 */
	public Tensor4D tanh() {
		return aplicar(x -> 2 / (1 + Math.exp(-2 * x)) - 1);
	}

	/**
	 * Aplica a função de ativação {@code Atan} (Arco Tangente)
	 * em todos os elementos do tensor.
	 * @return instância local alterada.
	 */
	public Tensor4D atan() {
		return aplicar(x -> Math.atan(x));
	}

	/**
	 * Calcula o valor {@code seno} de todos os elementos do tensor.
	 * @return instância local alterada.
	 */
	public Tensor4D sin() {
		return aplicar(x -> Math.sin(x));
	}

	/**
	 * Calcula o valor {@code cosseno} de todos os elementos do tensor.
	 * @return instância local alterada.
	 */
	public Tensor4D cos() {
		return aplicar(x -> Math.cos(x));
	}

	/**
	 * Calcula o valor {@code tangente} de todos os elementos do tensor.
	 * @return instância local alterada.
	 */
	public Tensor4D tan() {
		return aplicar(x -> Math.tan(x));
	}

	/**
	 * Calcula o valor {@code absoluto} de cada elemento do do tensor.
	 * @return instância local alterada.
	 */
	public Tensor4D abs() {
		return aplicar(x -> Math.abs(x));
	}

	/**
	 * Calcula o valor {@code exponencial} de cada elemento do do tensor.
	 * @return instância local alterada.
	 */
	public Tensor4D exp() {
		return aplicar(x -> Math.exp(x));
	}

	/**
	 * Calcula o valor {@code logaritmo natural} de cada elemento do do tensor.
	 * @return instância local alterada.
	 */
	public Tensor4D log() {
		return aplicar(x -> Math.log(x));
	}

	/**
	 * Monta as informações de exibição do tensor.
	 * @return string formatada.
	 */
	private String construirPrint() {
		final String identacao = " ".repeat(4);
        int tamMaximo = -1;
        for (double valor : dados) {
            int tamValor = String.format("%f", valor).length();
            if (tamValor > tamMaximo) tamMaximo = tamValor;
        }

        StringBuilder sb = new StringBuilder();

        int[] indices = new int[shape.length];
        boolean[] parentesisAbertos = new boolean[shape.length];

		sb.append(nome).append(" ").append(shapeStr()).append(" = [").append("\n");

		sb.append(identacao);
        for (int n = 0; n < tamanho(); n++) {
            for (int i = 0; i < indices.length; i++) {
                if (!parentesisAbertos[i]) {
                    sb.append("[");
                    parentesisAbertos[i] = true;
                }
            }

            final String valorStr = String.format("%f", get(indices));
            sb.append(" ".repeat(tamMaximo - valorStr.length()))
				.append(valorStr);

            final int idUltimaDim = shape.length - 1;
            if (indices[idUltimaDim] < shape[idUltimaDim] - 1) {
                sb.append(", ");
            }

            boolean qualquerParentesisAberto = false;
            int numParentesisFechados = 0;

            for (int i = indices.length - 1; i >= 0; i--) {
                indices[i] += 1;
                if (indices[i] >= shape[i]) {
                    indices[i] = 0;

                    sb.append("]");
                    if (i > 0 && indices[i - 1] < shape[i - 1] - 1) {
                        sb.append(",");
                    }

                    parentesisAbertos[i] = false;
                    qualquerParentesisAberto = true;
                    numParentesisFechados++;
                } else {
                    break;
                }
            }

            if (qualquerParentesisAberto) {
                if (numParentesisFechados > 1) {
                    sb.append("\n");
                }
                sb.append("\n").append(identacao);
                sb.append(" ".repeat(shape.length - numParentesisFechados));
            }
        }

		sb.delete(sb.length()-identacao.length()-1, sb.length()-1);
		sb.append("]").append("\n");

        return sb.toString().trim();
	}

	/**
	 * Exibe {@code via console} todo o conteúdo do tensor.
	 */
	public void print() {
		System.out.println(construirPrint());
	}

	/**
	 * Configura o nome do tensor.
	 * @param nome novo nome.
	 * @return instância local alterada.
	 */
	public Tensor4D nome(String nome) {
		if (nome != null) {
			nome = nome.trim();
			if (!nome.isEmpty()) this.nome = nome;
		}

		return this;
	}

	/**
	 * Retorna o nome do tensor.
	 * @return nome do tensor.
	 */
	public String nome() {
		return this.nome;
	}

	/**
	 * Retorna um array contendo as dimensões do tensor, seguindo a ordem:
	 * <pre>
	 *    dim = (d1, d2, d3, d4);
	 * </pre>
	 * @return dimensões do tensor.
	 */
	public int[] shape() {
		return shape.clone();
	}

	/**
	 * Retorna uma String contendo as dimensões do tensor, seguindo a ordem:
	 * <pre>
	 *    dim = (d1, d2, d3, d4);
	 * </pre>
	 * @return dimensões do tensor em formato de String.
	 */
	public String shapeStr() {
		StringBuilder sb = new StringBuilder();
		sb.append("(");
		for (int i = 0; i < shape.length-1; i++) {
			sb.append(shape[i]).append(", ");
		}
		sb.append(shape[shape.length-1]);
		sb.append(")");

		return sb.toString();
	}

	/**
	 * Retorna o {@code tamanho} da primeira dimensão do tensor.
	 * @return primeira dimensão do tensor.
	 */
	public int dim1() {
		return shape[0];
	}

	/**
	 * Retorna o {@code tamanho} da segunda dimensão do tensor.
	 * @return segunda dimensão do tensor.
	 */
	public int dim2() {
		return shape[1];
	}

	/**
	 * Retorna o {@code tamanho} da terceira dimensão do tensor.
	 * @return terceira dimensão do tensor.
	 */
	public int dim3() {
		return shape[2];
	}

	/**
	 * Retorna o {@code tamanho} da quarta dimensão do tensor.
	 * @return quarta dimensão do tensor.
	 */
	public int dim4() {
		return shape[3];
	}

	/**
	 * Verifica se o índice fornecido está dentro dos limites da dimensão desejada.
	 * <p>
	 * Exemplo:
	 * </p>
	 * <pre>
	 *tensor.dims = (2, 2);
	 *tensor.validarDimensao(0, tensor.dim4()) //true
	 *tensor.validarDimensao(2, tensor.dim4()) //false
	 * </pre>
	 * @param tDim dimensão correspondente do tensor.
	 * @param dim dimensão para ser avaliada.
	 * @return {@code true} caso o índice seja válido, {@code false} caso contrário.
	 */
	public boolean validarDimensao(int tDim, int dim) {
		return (dim >= 0) && (dim < tDim);
	}

	/**
	 * Verifica se o índice fornecido está dentro do alcance da primeira
	 * dimensão do tensor.
	 * @param id índice desejado.
	 */
	private void verificarIndiceD1(int id) {
		if (!validarDimensao(dim1(), id)) {
			throw new IllegalArgumentException(
				"\nÍndice da primeira dimensão (" + id + ") inválido."
			);
		}
	}

	/**
	 * Verifica se o índice fornecido está dentro do alcance da segunda
	 * dimensão do tensor.
	 * @param id índice desejado.
	 */
	private void verificarIndiceD2(int id) {
		if (!validarDimensao(dim2(), id)) {
			throw new IllegalArgumentException(
				"\nÍndice da segunda dimensão (" + id + ") inválido."
			);
		}
	}

	/**
	 * Verifica se o índice fornecido está dentro do alcance da terceira
	 * dimensão do tensor.
	 * @param id índice desejado.
	 */
	private void verificarIndiceD3(int id) {
		if (!validarDimensao(dim3(), id)) {
			throw new IllegalArgumentException(
				"\nÍndice da terceira dimensão (" + id + ") inválido."
			);
		}
	}

	/**
	 * Retorna a quantidade total de elementos no tensor.
	 * @return número elementos do tensor.
	 */
	public int tamanho() {
		return dados.length;
	}

	/**
	 * Retoran a quantidade de dimensões do tensor.
	 * @return quantidade de dimensões do tensor.
	 */
	public int numDim() {
		int dims = 0;

		for (int i = shape.length-1; i >= 0; i--) {
			if (shape[i] > 1) dims++;
		}

		return dims;
	}

	/**
	 * Clona o conteúdo do tensor numa instância separada.
	 * @return clone da instância local.
	 */
	public Tensor4D clone() {
		return new Tensor4D(this);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(construirPrint());

		int tamanho = sb.length();
		sb.deleteCharAt(tamanho - 1);// remover ultimo "\n"
		sb.deleteCharAt(tamanho - 2);// remover ultimo "\n"

		sb.append(" <tipo: ")
			.append(dados.getClass().getComponentType().getSimpleName())
			.append(">");
		
		sb.append(" <hash: ")
			.append(Integer.toHexString(hashCode()))
			.append(">");
		
		sb.append("]\n");

		return sb.toString();
	}

	@Override
	public boolean equals(Object obj) {
		return (obj instanceof Tensor4D) && comparar((Tensor4D) obj);
	}

	@Override
	public Iterator<Double> iterator() {
		return new Tensor4DIterator();
	}

	/**
	 * Iterador para usar com o tensor, usando para percorrer
	 * os elementos do tensor sequencialmente.
	 */
	class Tensor4DIterator implements Iterator<Double> {

		/**
		 * Contador do índice atual.
		 */
		private int indice = 0;

		@Override
		public boolean hasNext() {
			return indice < tamanho();
		}

		@Override
		public Double next() {
			return dados[indice++];
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException(
				"\nSem suporte."
			);
		}
	}
}
