package jnn.core.tensor;

import java.util.Arrays;
import java.util.Iterator;
import java.util.function.DoubleUnaryOperator;

import jnn.serializacao.SerialTensor;

import java.util.function.DoubleBinaryOperator;

/**
 * <h2>
 *		Tensor multidimensional
 * </h2>
 * Implementação de um array multidimensional com finalidade de simplificar 
 * o uso de estrutura de dados dentro da biblioteca.
 * <p>
 * 		O tensor possui algumas funções próprias com intuito de aproveitar a
 * 		velocidade de processamento usando um único array contendo seus dados.
 * </p>
 * <h2>
 *		Criação:
 * </h2>
 * <pre>
 *Tensor tensor = new Tensor(2, 2);
 *tensor = [
 *  [[0.0, 0.0],
 *   [0.0, 0.0]]
 *]
 * 
 *double[][] mat = { {...}, {...}, ... };
 *Tensor tensor = new Tensor(mat);
 * </pre>
 * <h2>
 *		Operações
 * </h2>
 *		Quase todas as operações realizadas acontecem localmente,
 *		alterando o conteúdo da instância que realizou o procedimento.
 * <pre>
 *Tensor a = [
 *	[1, 2, 3]
 *]
 *Tensor b = [
 *	[1, 2, 3]
 *]
 *a.add(b);// resultado em A
 * </pre>
 * @author Thiago Barroso, acadêmico de Engenharia da Computação pela
 * Universidade Federal do Pará, Campus Tucuruí. Maio/2024.
 */
public class Tensor implements Iterable<Variavel>, Cloneable {
    
	/**
	 * Dimensões do tensor.
	 */
	private int[] shape;

	/**
	 * Strides do tensor. Cada posição indica o salto no array de dados
	 * necessário para avançar uma unidade na respectiva dimensão.
	 */
	private int[] strides;

	/**
	 * Conjunto de elementos do tensor.
	 */
	private Variavel[] dados;

	/**
	 * Nome do tensor.
	 */
	private String nome = getClass().getSimpleName();

	/**
	 * Inicializa um tensor a partir de outra instância.
	 * <p>
	 *		O conteúdo do tensor recebido será copiado.
	 * </p>
	 * @param tensor tensor desejado.
	 */
    public Tensor(Tensor tensor) {
        this.shape = copiarShape(tensor.shape());

        int n = tensor.tam();
        dados = initDados(n);
		copiarElementos(tensor.dados);
		calcularStrides();
    }

	/**
	 * Inicializa um tensor a partir de um array 5D primitivo.
	 * @param elms elementos desejados.
	 */
	public Tensor(double[][][][][] elms) {
		if (elms == null) {
			throw new IllegalArgumentException(
				"\nTensor nulo."
			);
		}

		shape = copiarShape(
            elms.length, 
            elms[0].length, 
            elms[0][0].length, 
            elms[0][0][0].length,
            elms[0][0][0][0].length
        );

		dados = initDados(shape[0] * shape[1] * shape[2] * shape[3] * shape[4]);
		copiar(elms);
		calcularStrides();
	}

	/**
	 * Inicializa um tensor a partir de um array 4D primitivo.
	 * @param elms elementos desejados.
	 */
	public Tensor(double[][][][] elms) {
		if (elms == null) {
			throw new IllegalArgumentException(
				"\nTensor nulo."
			);
		}

		this.shape = new int[]{
            elms.length, 
            elms[0].length, 
            elms[0][0].length, 
            elms[0][0][0].length
        };

		dados = initDados(shape[0] * shape[1] * shape[2] * shape[3]);
		copiar(elms);
		calcularStrides();
	}

	/**
	 * Inicializa um tensor a partir de um array 3D primitivo.
	 * @param elms elementos desejados.
	 */
	public Tensor(double[][][] elms) {
		if (elms == null) {
			throw new IllegalArgumentException(
				"\nO tensor fornecido é nulo."
			);
		}

		this.shape = new int[]{
            elms.length, 
            elms[0].length, 
            elms[0][0].length,
        };

		dados = initDados(shape[0] * shape[1] * shape[2]);
		copiar(elms);
		calcularStrides();
	}

	/**
	 * Inicializa um tensor a partir de um array 2D primitivo.
	 * @param elms elementos desejados.
	 */
	public Tensor(double[][] elms) {
		if (elms == null) {
			throw new IllegalArgumentException(
				"\nA matriz fornecida é nula."
			);
		}

		int col = elms[0].length;
		for (int i = 1; i < elms.length; i++) {
			if (elms[i].length != col) {
				throw new IllegalArgumentException(
					"\nA matriz deve conter a mesma quantidade de linhas para todas as colunas."
				);
			}
		}

		this.shape = copiarShape(new int[]{elms.length, elms[0].length});
		dados = initDados(elms.length * elms[0].length);
		copiar(elms);
		calcularStrides();
	}

	/**
	 * Inicializar um tensor a partir de um conjunto de dados e formato
	 * pré-definidos.
	 * @param elms elementos desejados.
	 * @param shape formato desejado.
	 */
	public Tensor(double[] elms, int... shape) {
		int[] s  = copiarShape(shape);
		int tam = calcularTamanho(s);
		if (tam != elms.length) {
			throw new IllegalArgumentException(
				"\nTamanho dos dados (" + elms.length + ") não corresponde ao " +
				"formato fornecido (" + tam + ")"
			);
		}
		this.shape = s;

		this.dados = initDados(tam);
		for (int i = 0; i < tam; i++) {
			this.dados[i].set(elms[i]);
		}
		calcularStrides();
	}

    /**
     * Inicializa um novo tensor vazio a partir de um formato especificado.
     * @param shape formato desejado.
     */
    public Tensor(int... shape) {
        if (shape == null) {
            throw new IllegalArgumentException(
                "\nShape nulo."
            );
        }

        int tam = calcularTamanho(shape);

        this.shape = copiarShape(shape);
        dados = initDados(tam);
		calcularStrides();
    }

	/**
	 * Inicializa um tensor a partir de um array de variáveis.
	 * @param arr array desejado.
	 * @param dims dimensões desejadas.
	 */
    private Tensor(Variavel[] arr, int... dims) {
        shape = copiarShape(dims);
		calcularStrides();

		if (arr == null) {
			throw new IllegalArgumentException(
				"\nConjunto de elementos nulo"
			);
		}

		if (arr.length != calcularTamanho(dims)) {
			throw new IllegalArgumentException(
				"\nNúmero de elementos das dimensões dadas (" + calcularTamanho(dims) + "), " +
				"deve ser igual ao número de elementos dos dados fornecidos (" + arr.length + ")"
			);
		}

		dados = arr;
    }

	/**
	 * Auxiliar na inicialização do conjunto de dados do tensor.
	 * @param tamanho tamanho desejado.
	 * @return array de dados alocado.
	 */
	private Variavel[] initDados(int tamanho) {
		Variavel[] d = new Variavel[tamanho];
		for (int i = 0; i < tamanho; i++) {
			d[i] = new Variavel(0.0);
		}

		return d;
	}

	/**
	 * Calcula os strides do tensor a partir do shape. O último stride é 1
	 * (avanço unitário no array), e os anteriores são obtidos pelo produto
	 * acumulado dos tamanhos das dimensões seguintes.
	 */
	private void calcularStrides() {
		strides = new int[shape.length];
		strides[shape.length - 1] = 1;
		for (int i = shape.length - 2; i >= 0; i--) {
			strides[i] = strides[i + 1] * shape[i + 1];
		}
	}

	/**
	 * Calcula a quantidade de elementos de acordo com o formato informado.
	 * @param shape formato desejado.
	 * @return tamanho do array de elementos necessário.
	 */
    private int calcularTamanho(int[] shape) {
        if (shape.length == 0) return 0;

        int tam = 1;
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] < 1) {
                throw new IllegalArgumentException(
                    "\nArray de formato deve conter valores maiores que 1."
                );
            }

            tam *= shape[i];
        }

        return tam;
    }

    /**
     * Copia valores relevantes para o formato do tensor.
     * @param shape shape desejado.
     * @return shape com valores úteis.
     */
	private int[] copiarShape(int... shape) {
		if (shape.length == 0) {
			throw new IllegalArgumentException(
				"\nShape vazio."
			);
		}

		return shape.clone();
		
		//TODO reconsiderar isso aqui futuramente
		// int inicio = 0;
		// boolean difUmEncontrado = false;
		
		// for (int i = 0; i < shape.length; i++) {
		// 	if (shape[i] != 1) {
		// 		inicio = i;
		// 		difUmEncontrado = true;
		// 		break;
		// 	}
		// }
	
		// if (!difUmEncontrado) {
		// 	return new int[]{1};
		// }
	
		// int[] s = new int[shape.length - inicio];
		// System.arraycopy(shape, inicio, s, 0, s.length);
	
		// return s;
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
	 *    [[1, 2],
	 *     [3, 4]]
	 *]
	 *
	 *r = tensor.reshape(4);
	 *
	 *r = [
	 *    [1, 2, 3, 4]
	 *]
	 * </pre>
	 * @param dim array contendo as novas dimensões.
	 * @return instância local alterada.
	 */
	public Tensor reshape(int... shape) {
		int[] novoShape = copiarShape(shape);
		int novoTam = calcularTamanho(novoShape);

		if (tam() != novoTam) {
			throw new IllegalArgumentException(
				"\nQuatidade de elementos com as novas dimensões (" + novoTam +
				") deve ser igual a quantidade de elementos do tensor (" + tam() + ")."
			);
		}

		this.shape = novoShape;

		return this;
	}

	/**
	 * Cria uma nova visualização do tensor com as dimensões especificadas.
	 * @param shape dimensões desejadas.
	 * @return {@code Tensor} com a visualização desejada.
	 */
	public Tensor view(int... shape) {
		return new Tensor(dados, shape);
	}

	/**
	 * Transpõe o conteúdo do tensor.
	 * @return {@code Tensor} transposto.
	 */
    public Tensor transpor() {
        if (shape.length == 1) {
			//transpor tensor coluna
			Tensor t = new Tensor(shape[0], 1);
			t.copiarElementos(dados);
			return t;
        }
		
		if (shape.length == 2 && shape[1] == 1) {
			Tensor t = new Tensor(shape[0]);
			t.copiarElementos(dados);
			return t;
		}

        int[] novoShape = new int[shape.length];
        for (int i = 0; i < shape.length; i++) {
            novoShape[i] = shape[shape.length - i - 1];
        }
		
        Tensor t = new Tensor(novoShape);

        int[] idsOriginais = new int[shape.length];
        int[] idsTranspostos = new int[shape.length];
        for (int i = 0; i < dados.length; i++) {
            int temp = i;
            for (int j = shape.length - 1; j >= 0; j--) {
                idsOriginais[j] = temp % shape[j];
                temp /= shape[j];
            }

            for (int j = 0; j < shape.length; j++) {
                idsTranspostos[j] = idsOriginais[shape.length - j - 1];
            }

			int indiceTransposto = t.indice(idsTranspostos);

            t.dados[indiceTransposto] = dados[i];
        }

        return t;
    }

	/**
	 * Copia os elementos do tensores multiplas vezes.
	 * <p>
	 * 		Exemplo:
	 * </p>
	 * <pre>
	 *tensor = [1, 2, 3];
	 *bloco = tensor.bloco(3);
	 *bloco = [
	 *	[[1, 2, 3],
	 *	 [1, 2, 3],
	 *	 [1, 2, 3]]
	 *]
	 * </pre>
	 * @param n quantidade de repetições.
	 * @return {@code Tensor} com as modificações.
	 */
	public Tensor bloco(int n) {
		if (numDim() > 1) {
			throw new UnsupportedOperationException(
				"\nSem suporte para tensor com mais de uma dimensão."
			);
		}

		int elementos = tam();

		Variavel[] arr = new Variavel[elementos * n];
		for (int i = 0; i < n; i++) {
			int inicio = i*elementos;
			for (int j = 0; j < elementos; j++) {
				arr[inicio + j] = new Variavel(dados[j]);
			}
		}

		Tensor bloco = new Tensor(arr, arr.length);
		bloco.reshape(n, elementos);

		return bloco;
	}

    /**
     * Calcula o índice de um elementos dentro do conjunto de dados do tensor.
     * @param ids índices desejados.
     * @return índice correspondente no array de elementos do tensor.
     */
    private int indice(int... ids) {
        // if (numDim() != ids.length) {
        //     throw new IllegalArgumentException(
		// 		"\nNúmero de dimensões fornecidas " + ids.length + 
		// 		" não corresponde às " + numDim() + " do tensor."
		// 	);
        // }
    
        // int id = 0;
        // int multiplicador = 1;
    
        // for (int i = shape.length - 1; i >= 0; i--) {
        //     if (ids[i] < 0 || ids[i] >= shape[i]) {
        //         throw new IllegalArgumentException(
		// 			"\nÍndice " + ids[i] + " fora dos limites para a dimensão " + i +
		// 			" (tamanho = " + shape[i] + ");"
		// 		);
        //     }
        //     id += ids[i] * multiplicador;
        //     multiplicador *= shape[i];
        // }
    
        // return id;

		if (numDim() != ids.length) {
			throw new IllegalArgumentException(
				"Número de dimensões fornecidas " + ids.length +
				" não corresponde às " + numDim() + " do tensor."
			);
		}

		int id = 0;
		for (int i = 0; i < ids.length; i++) {
			if (ids[i] < 0 || ids[i] >= shape[i]) {
				throw new IllegalArgumentException(
					"Índice " + ids[i] + " fora dos limites para a dimensão " + i +
					" (tamanho = " + shape[i] + ");"
				);
			}
			id += ids[i] * strides[i];
		}
		return id;
    }

	/**
	 * Retorna o elemento do tensor de acordo com os índices fornecidos.
	 * @param ids índices desejados para busca.
	 * @return valor de acordo com os índices.
	 */
    public double get(int... ids) {
        return dados[indice(ids)].get();
    }

	/**
	 * Edita o valor do tensor usando o valor informado.
	 * @param x valor desejado.
	 * @param ids índices para atribuição.
	 */
    public void set(Number x, int... ids) {
        dados[indice(ids)].set(x);
    }

	/**
	 * Edita o valor do tensor usando uma variável.
	 * @param x variável com valor desejado.
	 * @param ids índices para atribuição.
	 */
	public void set(Variavel x, int... ids) {
		dados[indice(ids)].set(x);
	}

	/**
	 * Preenche todo o conteúdo do tensor com o valor fornecido.
	 * @param x valor desejado.
	 * @return instância local alterada.
	 */
	public Tensor preencher(Number x) {
		final int n = tam();
		for (int i = 0; i < n; i++) {
			dados[i].set(x);
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
	public Tensor preencherContador(boolean cres) {
		int tam = tam();

		if (cres) {
			for (int i = 0; i < tam; i++) {
				dados[i].set(i+1);
			}

		} else {
			for (int i = 0; i < tam; i++) {
				dados[i].set(tam-i-1);
			}
		}

		return this;
	}

	/**
	 * Zera todo o conteúdo o tensor.
	 * @return instância local alterada.
	 */
	public Tensor zero() {
		for (Variavel x : dados) {
			x.zero();
		}

		return this;
	}

	/**
	 * Copia todo o conteúdo do tensor na instância local.
	 * @param tensor {@code Tensor} desejado.
	 * @param tensor {@code Tensor} desejado.
	 * @return instância local alterada.
	 */
	public Tensor copiar(Tensor tensor) {
		if (!compararShape(tensor)) {
			throw new IllegalArgumentException(
				"\nDimensões " + shapeStr() + " incompatíveis com as do" +
				" tensor recebido " + tensor.shapeStr()
			);
		}

		final int n = tam();
		for (int i = 0; i < n; i++) {
			dados[i].set(tensor.dados[i]);
		}

		return this;
	}

	/**
	 * Copia todo o conteúdo do array na instância local.
	 * @param arr array desejado.
	 * @return instância local alterada.
	 */
    public Tensor copiar(double[][][][][] arr) {
        if (numDim() != 5) {
            throw new IllegalArgumentException(
                "\nTensor tem " + numDim() + " dimensões, mas deve" +
                " ter 5."
            );
        }

        int d1 = shape[0];
        int d2 = shape[1];
        int d3 = shape[2];
        int d4 = shape[3];
        int d5 = shape[4];

		if ((d1 != arr.length) ||
			(d2 != arr[0].length) ||
			(d3 != arr[0][0].length) ||
			(d4 != arr[0][0][0].length) ||
			(d5 != arr[0][0][0][0].length)) {
			throw new IllegalArgumentException(
				"\nDimensões do tensor " + shapeStr() +
				" incompatíveis com as do array recebido ("
				+ arr.length + ", " 
				+ arr[0].length + ", "
				+ arr[0][0].length + ", "
				+ arr[0][0][0].length + ", "
				+ arr[0][0][0][0].length + ")."
			);
		}

		int cont = 0;
		for (int i = 0; i < d1; i++) {
			for (int j = 0; j < d2; j++) {
				for (int k = 0; k < d3; k++) {
					for (int l = 0; l < d4; l++) {
						for (int m = 0; m < d5; m++) {
							dados[cont++].set(arr[i][j][k][l][m]);
						}
					}
				}
			}
		}

		return this;
    }

	/**
	 * Copia todo o conteúdo do array na instância local.
	 * @param arr array desejado.
	 * @return instância local alterada.
	 */
    public Tensor copiar(double[][][][] arr) {
        if (numDim() != 4) {
            throw new IllegalArgumentException(
                "\nTensor tem " + numDim() + " dimensões, mas deve" +
                " ter 4."
            );
        }

        int d1 = shape[0];
        int d2 = shape[1];
        int d3 = shape[2];
        int d4 = shape[3];

		if ((d1 != arr.length) ||
			(d2 != arr[0].length) ||
			(d3 != arr[0][0].length) ||
			(d4 != arr[0][0][0].length)) {
			throw new IllegalArgumentException(
				"\nDimensões do tensor " + shapeStr() +
				" incompatíveis com as do array recebido ("
				+ arr.length + ", " + arr[0].length + ", " + arr[0][0].length + ", " + arr[0][0][0].length
				+ ")."
			);
		}

		int cont = 0;
		for (int i = 0; i < d1; i++) {
			for (int j = 0; j < d2; j++) {
				for (int k = 0; k < d3; k++) {
					for (int l = 0; l < d4; l++) {
						dados[cont++].set(arr[i][j][k][l]);
					}
				}
			}
		}

		return this;
    }

	/**
	 * Copia todo o conteúdo do array na instância local.
	 * @param arr array desejado.
	 * @return instância local alterada.
	 */
    public Tensor copiar(double[][][] arr) {
        if (numDim() != 3) {
            throw new IllegalArgumentException(
                "\nTensor tem " + numDim() + " dimensões, mas deve" +
                " ter 3."
            );
        }

        int d1 = shape[0];
        int d2 = shape[1];
        int d3 = shape[2];

		if ((d1 != arr.length) ||
			(d2 != arr[0].length) ||
			(d3 != arr[0][0].length)) {
			throw new IllegalArgumentException(
				"\nDimensões do tensor " + shapeStr() +
				" incompatíveis com as do array recebido ("
				+ arr.length + ", " + arr[0].length + ", " + arr[0][0].length + ")."
			);
		}

		int cont = 0;
		for (int i = 0; i < d1; i++) {
			for (int j = 0; j < d2; j++) {
				for (int k = 0; k < d3; k++) {
					dados[cont++].set(arr[i][j][k]);
				}
			}
		}

		return this;
    }

	/**
	 * Copia todo o conteúdo do array na instância local.
	 * @param arr array desejado.
	 * @return instância local alterada.
	 */
    public Tensor copiar(double[][] arr) {
        if (numDim() > 2) {
            throw new IllegalArgumentException(
                "\nTensor tem " + numDim() + " dimensões, mas deve" +
                " ter 2."
            );
        }

        int lin = (numDim() == 1) ? 1: shape[0];
        int col = (numDim() == 1) ? shape[0] : shape[1];

		if ((lin != arr.length) ||
			(col != arr[0].length)) {
			throw new IllegalArgumentException(
				"\nDimensões do tensor " + shapeStr() +
				" incompatíveis com as do array recebido ("
				+ arr.length + ", " + arr[0].length + ")."
			);
		}

		int id = 0;
		for (int i = 0; i < lin; i++) {
			for (int j = 0; j < col; j++) {
				dados[id++].set(arr[i][j]);
			}
		}

		return this;
    }

	/**
	 * Copia todo o conteúdo do array na instância local.
	 * @param arr array desejado.
	 * @return instância local alterada.
	 */
    public Tensor copiar(double[] arr) {
        if (numDim() != 1) {
            throw new IllegalArgumentException(
                "\nTensor tem " + numDim() + " dimensões, mas deve" +
                " ter 1."
            );
        }

		if ((tam() != arr.length)) {
			throw new IllegalArgumentException(
				"\nDimensões do tensor " + shapeStr() +
				" incompatíveis com as do array recebido (" + arr.length + ")."
			);
		}

		for (int i = 0; i < arr.length; i++) {
			dados[i].set(arr[i]);
		}

		return this;
    }

	/**
	 * Copia apenas os dados contidos no tensor, sem levar em consideração 
	 * suas dimensões.
	 * <p>
	 *		Ainda é necessário que a quantidade de elementos de ambos os 
	 *		tensores sejam iguais.
	 * </p>
	 * @param tensor {@code Tensor} desejado para cópia.
	 * @return instância local alterada.
	 */
	public Tensor copiarElementos(Tensor tensor) {
		if (tam() != tensor.tam()) {
			throw new IllegalArgumentException(
				"\nOs tensores devem conter o mesmo número de elementos. Local = " + tam() + 
				"e recebido = " + tensor.tam()
			);
		}

		final int n = tam();
		for (int i = 0; i < n; i++) {
			dados[i].set(tensor.dados[i]);
		}

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
	public Tensor copiarElementos(Variavel[] elementos) {
		if (elementos.length != tam()) {
			throw new IllegalArgumentException(
				"\nTamanho do array fornecido (" + elementos.length + ") inconpatível" +
				"com os elementos do tensor (" + tam() + ")."
			);
		}

		final int n = tam();
		for (int i = 0; i < n; i++) {
			dados[i].set(elementos[i]);
		}

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
	public Tensor copiarElementos(double[] elementos) {
		if (elementos.length != tam()) {
			throw new IllegalArgumentException(
				"\nTamanho do array fornecido (" + elementos.length + ") inconpatível" +
				"com os elementos do tensor (" + tam() + ")."
			);
		}

		for (int i = 0; i < elementos.length; i++) {
			dados[i].set(elementos[i]);
		}
		
		return this;
	}

	/**
	 * Adiciona o valor informado em todos os elementos do tensor.
	 * @param x valor desejado.
	 * @return instância local alterada.
	 */
	public Tensor add(Number x) {
		final int n = tam();
		for (int i = 0; i < n; i++) {
			dados[i].add(x);
		}

		return this;
	}

	/**
	 * Adiciona o valor da variável informada em todos os elementos do tensor.
	 * @param x valor desejado.
	 * @return instância local alterada.
	 */
	public Tensor add(Variavel x) {
		return add(x.get());
	}

	/**
	 * Adiciona todo o conteúdo {@code elemento a elemento} usando o tensor recebido,
	 * seguindo a expressão:
	 * <pre>
	 *  this += tensor
	 * </pre>
	 * @param tensor {@code Tensor} com conteúdo.
	 * @return instância local alterada.
	 */
    public Tensor add(Tensor tensor) {
        if (!compararShape(tensor)) {
            throw new IllegalArgumentException(
                "\nTensor fornecido possui shape " + tensor.shapeStr() +
				", shape esperado " + shapeStr()
            );
        }

        int n = tam();
        for (int i = 0; i < n; i++) {
            dados[i].add(tensor.dados[i]);
		}

        return this;
    }

	/**
	 * Adiciona localmente o resultado da adição elemento a elemento entre 
	 * os tensores A e B.
	 * <p>
	 * 		Podendo ser expresso por:
	 * </p>
	 * <pre>
	 *		this += (a + b)
	 * </pre>
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @return instância local alterada.
	 */
    public Tensor addSoma(Tensor a, Tensor b) {
		return aplicar(
			this, a, b, 
			(t, t1, t2) -> t += (t1 + t2)
		);
    }

	/**
	 * Adiciona localmente o resultado da diferença elemento a elemento entre 
	 * os tensores A e B.
	 * <p>
	 * 		Podendo ser expresso por:
	 * </p>
	 * <pre>
	 *		this += (a - b)
	 * </pre>
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @return instância local alterada.
	 */
    public Tensor addSub(Tensor a, Tensor b) {
		return aplicar(
			this, a, b, 
			(t, t1, t2) -> t += (t1 - t2)
		);
    }

	/**
	 * Adiciona localmente o resultado da multiplicação elemento a elemento entre 
	 * os tensores A e B.
	 * <p>
	 * 		Podendo ser expresso por:
	 * </p>
	 * <pre>
	 *		this += (a * b)
	 * </pre>
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @return instância local alterada.
	 */
    public Tensor addMul(Tensor a, Tensor b) {
		return aplicar(
			this, a, b, 
			(t, t1, t2) -> t += (t1 * t2)
		);
    }

	/**
	 * Adiciona localmente o resultado da divisão elemento a elemento entre 
	 * os tensores A e B.
	 * <p>
	 * 		Podendo ser expresso por:
	 * </p>
	 * <pre>
	 *		this += (a / b)
	 * </pre>
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @return instância local alterada.
	 */
    public Tensor addDiv(Tensor a, Tensor b) {
		return aplicar(
			this, a, b, 
			(t, t1, t2) -> t += (t1 / t2)
		);
    }

	/**
	 * Adiciona o valor informado ao conteúdo do tensor.
	 * @param x valor desejado.
	 * @param ids índices desejados para adição.
	 * @return instância local alterada.
	 */
	public Tensor add(Number x, int... ids) {
		dados[indice(ids)].add(x);
		return this;
	}

	/**
	 * Subtrai o valor informado em todos os elementos do tensor.
	 * @param x valor desejado.
	 * @return instância local alterada.
	 */
	public Tensor sub(Number x) {
		final int n = tam();
		for (int i = 0; i < n; i++) {
			dados[i].sub(x);
		}

		return this;
	}

	/**
	 * Subtrai o valor da variável informada em todos os elementos do tensor.
	 * @param x valor desejado.
	 * @return instância local alterada.
	 */
	public Tensor sub(Variavel x) {
		return sub(x.get());
	}

	/**
	 * Subtrai todo o conteúdo {@code elemento a elemento} usando o tensor recebido,
	 * seguindo a expressão:
	 * <pre>
	 *  this -= tensor
	 * </pre>
	 * @param tensor {@code Tensor} com conteúdo.
	 * @return instância local alterada.
	 */
    public Tensor sub(Tensor tensor) {
        if (!compararShape(tensor)) {
            throw new IllegalArgumentException(
                "\nTensor fornecido deve conter o mesmo shape."
            );
        }

        int n = tam();
        for (int i = 0; i < n; i++) {
            dados[i].sub(tensor.dados[i]);
        }

        return this;
    }

	/**
	 * Subtrai localmente o resultado da adição elemento a elemento entre 
	 * os tensores A e B.
	 * <p>
	 * 		Podendo ser expresso por:
	 * </p>
	 * <pre>
	 *		this -= (a + b)
	 * </pre>
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @return instância local alterada.
	 */
    public Tensor subSoma(Tensor a, Tensor b) {
		return aplicar(
			this, a, b, 
			(t, t1, t2) -> t -= (t1 + t2)
		);
    }

	/**
	 * Subtrai localmente o resultado da diferença elemento a elemento entre 
	 * os tensores A e B.
	 * <p>
	 * 		Podendo ser expresso por:
	 * </p>
	 * <pre>
	 *		this -= (a - b)
	 * </pre>
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @return instância local alterada.
	 */
    public Tensor subSub(Tensor a, Tensor b) {
		return aplicar(
			this, a, b, 
			(t, t1, t2) -> t -= (t1 - t2)
		);
    }

	/**
	 * Subtrai localmente o resultado da multiplicação elemento a elemento entre 
	 * os tensores A e B.
	 * <p>
	 * 		Podendo ser expresso por:
	 * </p>
	 * <pre>
	 *		this -= (a * b)
	 * </pre>
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @return instância local alterada.
	 */
    public Tensor subMul(Tensor a, Tensor b) {
		return aplicar(
			this, a, b, 
			(t, t1, t2) -> t -= (t1 * t2)
		);
    }

	/**
	 * Subtrai localmente o resultado da divisão elemento a elemento entre 
	 * os tensores A e B.
	 * <p>
	 * 		Podendo ser expresso por:
	 * </p>
	 * <pre>
	 *		this -= (a / b)
	 * </pre>
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @return instância local alterada.
	 */
    public Tensor subDiv(Tensor a, Tensor b) {
		return aplicar(
			this, a, b, 
			(t, t1, t2) -> t -= (t1 / t2)
		);
    }

	/**
	 * Subtrai o valor informado ao conteúdo do tensor.
	 * @param x valor desejado.
	 * @param ids índices desejados para adição.
	 * @return instância local alterada.
	 */
	public Tensor sub(Number x, int... ids) {
		dados[indice(ids)].sub(x);
		return this;
	}

	/**
	 * Multiplica o valor informado em todos os elementos do tensor.
	 * @param x valor desejado.
	 * @return instância local alterada.
	 */
	public Tensor mul(Number x) {
		final int n = tam();
		for (int i = 0; i < n; i++) {
			dados[i].mul(x);
		}

		return this;
	}

	/**
	 * Multiplica o valor da variável informada em todos os elementos do tensor.
	 * @param x valor desejado.
	 * @return instância local alterada.
	 */
	public Tensor mul(Variavel x) {
		return mul(x.get());
	}
	
	/**
	 * Multiplica todo o conteúdo {@code elemento a elemento} usando o tensor recebido,
	 * seguindo a expressão:
	 * <pre>
	 *  this *= tensor
	 * </pre>
	 * @param tensor {@code Tensor} com conteúdo.
	 * @return instância local alterada.
	 */
    public Tensor mul(Tensor tensor) {
        if (!compararShape(tensor)) {
            throw new IllegalArgumentException(
                "\nTensor fornecido deve conter o mesmo shape."
            );
        }

        int n = tam();
        for (int i = 0; i < n; i++) {
            dados[i].mul(tensor.dados[i]);
        }

        return this;
    }

	/**
	 * Multiplica localmente o resultado da adição elemento a elemento entre 
	 * os tensores A e B.
	 * <p>
	 * 		Podendo ser expresso por:
	 * </p>
	 * <pre>
	 *		this *= (a + b)
	 * </pre>
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @return instância local alterada.
	 */
    public Tensor mulSoma(Tensor a, Tensor b) {
		return aplicar(
			this, a, b, 
			(t, t1, t2) -> t *= (t1 + t2)
		);
    }

	/**
	 * Multiplica localmente o resultado da subtração elemento a elemento entre 
	 * os tensores A e B.
	 * <p>
	 * 		Podendo ser expresso por:
	 * </p>
	 * <pre>
	 *		this *= (a - b)
	 * </pre>
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @return instância local alterada.
	 */
    public Tensor mulSub(Tensor a, Tensor b) {
		return aplicar(
			this, a, b, 
			(t, t1, t2) -> t *= (t1 - t2)
		);
    }

	/**
	 * Multiplica localmente o resultado da multiplicação elemento a elemento entre 
	 * os tensores A e B.
	 * <p>
	 * 		Podendo ser expresso por:
	 * </p>
	 * <pre>
	 *		this *= (a * b)
	 * </pre>
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @return instância local alterada.
	 */
    public Tensor mulMul(Tensor a, Tensor b) {
		return aplicar(
			this, a, b, 
			(t, t1, t2) -> t *= (t1 * t2)
		);
    }

	/**
	 * Multiplica localmente o resultado da divisão elemento a elemento entre 
	 * os tensores A e B.
	 * <p>
	 * 		Podendo ser expresso por:
	 * </p>
	 * <pre>
	 *		this *= (a * b)
	 * </pre>
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @return instância local alterada.
	 */
    public Tensor mulDiv(Tensor a, Tensor b) {
		return aplicar(
			this, a, b, 
			(t, t1, t2) -> t *= (t1 / t2)
		);
    }

	/**
	 * Multiplica o valor informado ao conteúdo do tensor.
	 * @param x valor desejado.
	 * @param ids índices desejados para adição.
	 * @return instância local alterada.
	 */
	public Tensor mul(Number x, int... ids) {
		dados[indice(ids)].mul(x);
		return this;
	}

	/**
	 * Divide o valor informado em todos os elementos do tensor.
	 * @param x valor desejado.
	 * @return instância local alterada.
	 */
	public Tensor div(Number x) {
		final int n = tam();
		for (int i = 0; i < n; i++) {
			dados[i].div(x);
		}

		return this;
	}

	/**
	 * Divide o valor da variável informada em todos os elementos do tensor.
	 * @param x valor desejado.
	 * @return instância local alterada.
	 */
	public Tensor div(Variavel x) {
		return div(x.get());
	}

	/**
	 * Divide todo o conteúdo {@code elemento a elemento} usando o tensor recebido,
	 * seguindo a expressão:
	 * <pre>
	 *  this /= tensor
	 * </pre>
	 * @param tensor {@code Tensor} com conteúdo.
	 * @return instância local alterada.
	 */
    public Tensor div(Tensor tensor) {
        if (!compararShape(tensor)) {
            throw new IllegalArgumentException(
                "\nTensor fornecido deve conter o mesmo shape."
            );
        }

        int n = tam();
        for (int i = 0; i < n; i++) {
            dados[i].div(tensor.dados[i]);
        }

        return this;
    }

	/**
	 * Divide localmente o resultado da adição elemento a elemento entre 
	 * os tensores A e B.
	 * <p>
	 * 		Podendo ser expresso por:
	 * </p>
	 * <pre>
	 *		this /= (a + b)
	 * </pre>
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @return instância local alterada.
	 */
    public Tensor divSoma(Tensor a, Tensor b) {
		return aplicar(
			this, a, b, 
			(t, t1, t2) -> t /= (t1 + t2)
		);
    }

	/**
	 * Divide localmente o resultado da subtração elemento a elemento entre 
	 * os tensores A e B.
	 * <p>
	 * 		Podendo ser expresso por:
	 * </p>
	 * <pre>
	 *		this /= (a - b)
	 * </pre>
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @return instância local alterada.
	 */
    public Tensor divSub(Tensor a, Tensor b) {
		return aplicar(
			this, a, b, 
			(t, t1, t2) -> t /= (t1 - t2)
		);
    }

	/**
	 * Divide localmente o resultado da multiplicação elemento a elemento entre 
	 * os tensores A e B.
	 * <p>
	 * 		Podendo ser expresso por:
	 * </p>
	 * <pre>
	 *		this /= (a * b)
	 * </pre>
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @return instância local alterada.
	 */
    public Tensor divMul(Tensor a, Tensor b) {
		return aplicar(
			this, a, b, 
			(t, t1, t2) -> t /= (t1 * t2)
		);
    }

	/**
	 * Divide localmente o resultado da divisão elemento a elemento entre 
	 * os tensores A e B.
	 * <p>
	 * 		Podendo ser expresso por:
	 * </p>
	 * <pre>
	 *		this /= (a / b)
	 * </pre>
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @return instância local alterada.
	 */
    public Tensor divDiv(Tensor a, Tensor b) {
		return aplicar(this, a, b, 
			(t, t1, t2) -> t /= (t1 / t2)
		);
    }

	/**
	 * Divide o valor contido no tensor pelo valor informado.
	 * @param x valor desejado.
	 * @param ids índices desejados para adição.
	 * @return instância local alterada.
	 */
	public Tensor div(Number x, int... ids) {
		dados[indice(ids)].div(x);
		return this;
	}

	/**
	 * Remove a dimensão desejada caso possua tamanho = 1.
	 * @param dim índice da dimensão desejada.
	 * @return instância local, talvez alterada.
	 */
	public Tensor squeeze(int dim) {
		if (dim < 0 || dim >= shape.length) {
			throw new IllegalArgumentException("\nDimensão " + dim + " inválida");
		}

		if (numDim() == 1) return this;// não fazer nada com tensores escalares
	
		if (shape[dim] != 1) {
			return this;// não alterar dimensões com tamanho != 1
		}
	
		int[] novoShape = new int[shape.length - 1];
		int id = 0;
		for (int i = 0; i < shape.length; i++) {
			if (i != dim) novoShape[id++] = shape[i];
		}
	
		shape = novoShape;
	
		return this;
	}

	/**
	 * Adiciona uma nova dimensão com tamanho = 1.
	 * @param dim índice da dimensão que será adicionada.
	 * @return instância local alterada.
	 */
    public Tensor unsqueeze(int dim) {
        if (dim < 0 || dim > shape.length) {
            throw new IllegalArgumentException(
				"\nDimensão " + dim + " inválida"
			);
        }
        
		final int n = numDim();
        
        int[] novoShape = new int[n + 1];
		for (int i = 0; i < dim; i++) {
            novoShape[i] = shape[i];
        }
        novoShape[dim] = 1;
        for (int i = dim; i < n; i++) {
            novoShape[i + 1] = shape[i];
        }

        this.shape = novoShape;

		return this;
    }

	/**
	 * Achata os dados do tensor.
	 * @return instância local alterada.
	 */
	public Tensor flatten() {
		shape = new int[] { tam() };
		return this;
	}

	/**
	 * Fatia o conteúdo do tensor de acordo com os índices especificados.
	 * <p>
	 *		Exemplo:
	 * </p>
	 * <pre>
	 *tensor [
	 *	[[1, 2, 3],
	 *	 [4, 5, 6]]
	 *]
	 *
	 *slice = tensor.slice(new int[]{0, 0}, new int[]{1, 3});
	 * 
	 *slice = [
	 * 	[[1, 2, 3]]
	 *]
	 * </pre>
	 * @param idsInicio índices de incio do fatiamento (inclusivo).
	 * @param idsFim índices do fim do fatiamento (exclusivos).
	 * @return {@code Tensor} fatiado.
	 */
	public Tensor slice(int[] idsInicio, int[] idsFim) {
		if (idsInicio.length != shape.length || idsFim.length != shape.length) {
			throw new IllegalArgumentException(
				"\nNúmero de índices de início/fim não corresponde às dimensões do tensor (" + numDim() + ")."
			);
		}
	
		int nDims = numDim();
		int[] novoShape = new int[nDims];
		for (int i = 0; i < nDims; i++) {
			if (idsInicio[i] < 0 || idsInicio[i] >= shape[i] ||
				idsFim[i] < 0 || idsFim[i] > shape[i] || idsFim[i] <= idsInicio[i]) {
				throw new IllegalArgumentException(
					"\nÍndices de início/fim inválidos para a dimensão " + i
				);
			}
			novoShape[i] = idsFim[i] - idsInicio[i];
		}
	
		Tensor slice = new Tensor(novoShape);
	
		final int tam = tam();
		int[] indices = new int[nDims];
		int[] idsSlice = new int[nDims];
		boolean dentroSlice;
		int i, j;
		for (i = 0; i < tam; i++) {
			int id = i;
			for (j = nDims - 1; j >= 0; j--) {
				indices[j] = id % shape[j];
				id /= shape[j];
			}
	
			dentroSlice = true;
			for (j = 0; j < nDims; j++) {
				if (indices[j] < idsInicio[j] || indices[j] >= idsFim[j]) {
					dentroSlice = false;
					break;
				}
			}
	
			if (dentroSlice) {
				for (j = 0; j < nDims; j++) {
					idsSlice[j] = indices[j] - idsInicio[j];
				}

				//por padrão compartilhar as mesma variáveis
				slice.dados[indice(idsSlice)] = dados[indice(indices)];
			}
		}
	
		return slice;
	}

	/**
	 * Aplica a função recebida em todos os elementos do tensor.
	 * <p>
	 *      Exemplo:
	 * </p>
	 * <pre>
	 * tensor.aplicar(x -> Math.random());
	 * </pre>
	 * Onde {@code x} representa cada elemento dentro do tensor.
	 * 
	 * @param fun função desejada.
	 * @return instância local alterada.
	 */
    public Tensor aplicar(DoubleUnaryOperator fun) {
		final int n = tam();
		for (int i = 0; i < n; i++) {
			dados[i].set(fun.applyAsDouble(dados[i].get()));
		}

		return this;
	}

	/**
	 * Aplica a função recebida em todos os elementos do tensor.
	 * <p>
	 *      Exemplo:
	 * </p>
	 * <pre>
	 * tensor.aplicar(x -> Math.random());
	 * </pre>
	 * Onde {@code x} representa cada elemento dentro do tensor fornecido.
	 * @param tensor {@code Tensor} base.
	 * @param fun função para aplicar no tensor base.
	 * @return instância local alterada.
	 */
    public Tensor aplicar(Tensor tensor, DoubleUnaryOperator fun) {
		if (!compararShape(tensor)) {
			throw new IllegalArgumentException(
				"\nAs dimensões do tensor fornecido " + tensor.shapeStr() +
				" e as da instância local " + shapeStr() + " devem ser iguais."
			);
		}

		final int n = tam();
		for (int i = 0; i < n; i++) {
			dados[i].set(
				fun.applyAsDouble(tensor.dados[i].get())
			);
		}

		return this;
	}

	/**
	 * Aplica a função recebida em todos os elementos do tensor de acordo com a operação
	 * entre A e B.
	 * <p>
	 *      Exemplo:
	 * </p>
	 * <pre>
	 *Tensor a = new Tensor(2, 2);
	 *Tensor b = new Tensor(2, 2);
	 *Tensor c = new Tensor(2, 2);
	 *c.aplicar(a, b, (x, y) -> x + y);
	 * </pre>
	 * Onde:
	 * <p>{@code x} representa cada elemento dentro do tensor A.
	 * <p>{@code y} representa cada elemento dentro do tensor B.
	 * <p>
	 *		É necessário que todos os tensores possuam o mesmo formato.
	 * </p>
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @param fun função para aplicar no tensor local.
	 * @return instância local alterada.
	 */
    public Tensor aplicar(Tensor a, Tensor b, DoubleBinaryOperator fun) {
		if (!compararShape(a) || !compararShape(b)) {
			throw new IllegalArgumentException(
				"\nAs dimensões dos tensores A " + a.shapeStr() +
				", B " + b.shapeStr() +
				" e as da instância local " + shapeStr() + " devem ser iguais."
			);
		}
		
		final int n = tam();
		for (int i = 0; i < n; i++) {
			dados[i].set(
				fun.applyAsDouble(
					a.dados[i].get(),
					b.dados[i].get()
				)
			);
		}

		return this;
	}

	/**
	 * Aplica a função recebida em todos os elementos do tensor de acordo com a operação
	 * entre A, B e C.
	 * <p>
	 *      Exemplo:
	 * </p>
	 * <pre>
	 *Tensor a = new Tensor(2, 2);
	 *Tensor b = new Tensor(2, 2);
	 *Tensor c = new Tensor(2, 2);
	 *Tensor d = new Tensor(2, 2);
	 *d.aplicar(a, b, c, (x, y, z) -> x + y + z);
	 * </pre>
	 * Onde:
	 * <p>{@code x} representa cada elemento dentro do tensor A.
	 * <p>{@code y} representa cada elemento dentro do tensor B.
	 * <p>{@code z} representa cada elemento dentro do tensor C.
	 * <p>
	 *		É necessário que todos os tensores possuam o mesmo formato.
	 * </p>
	 * @param a {@code Tensor} A.
	 * @param b {@code Tensor} B.
	 * @param c {@code Tensor} C.
	 * @param fun função para aplicar no tensor local.
	 * @return instância local alterada.
	 */
	public Tensor aplicar(Tensor a, Tensor b, Tensor c, DoubleTernaryOperator fun) {
		if (!compararShape(a) || !compararShape(b) || !compararShape(c)) {
			throw new IllegalArgumentException(
				"\nAs dimensões dos tensores A " + a.shapeStr() +
				", B " + b.shapeStr() +
				" e C " + c.shapeStr() +
				" e as da instância local " + shapeStr() + " devem ser iguais."
			);
		}

		final int n = tam();
		for (int i = 0; i < n; i++) {
			dados[i].set(
				fun.applyAsDouble(
					a.dados[i].get(),
					b.dados[i].get(),
					c.dados[i].get()
				)
			);
		}
		
		return this;
	}

	/**
	 * Retorna o valor contido no tensor, caso ele possua apenas um elemento.
	 * @return valor contido no tensor.
	 */
	public double item() {
		if (tam() > 1) {
			throw new IllegalArgumentException(
				"\nO tensor deve conter apenas um elemento."
			);
		}

		return dados[0].get();
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
	public Tensor map(DoubleUnaryOperator fun) {
		Tensor t = new Tensor(shape());
		t.aplicar(this, fun);
		
		return t;
	}

	/**
	 * Aplica a função recebida em todos os elementos do tensor.
	 * <p>
	 *		Exemplo:
	 * </p>
	 * <pre>
	 *a = {1, 2, 3};
	 *b = {1, 2, 3};
	 *
	 *r = a.map(b, (x, y) -> x+y);
	 *r = {2, 4, 6};
	 *  </pre>
	 * Onde:
	 *{@code x} representa cada elemento dentro do tensor local.
	 *{@code y} representa cada elemento dentro do tensor fornecido.
	 * @param tensor segundo {@code Tensor} para aplicar a função.
	 * @param fun função desejada.
	 * @return novo {@code Tensor} contendo o resultado.
	 */
	public Tensor map(Tensor tensor, DoubleBinaryOperator fun) {
		if (!compararShape(tensor)) {
			throw new IllegalArgumentException(
				"\nTensor " + tensor.shapeStr() + " deve conter mesmo formato do " +
				"tensor local " + shapeStr()
			);
		}

		Tensor t = new Tensor(shape());
		t.aplicar(this, tensor, fun);

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
	public Tensor reduce(Number in, DoubleBinaryOperator fun) {
		Variavel res = new Variavel(in);
		for (Variavel val : dados) {
			res.set(fun.applyAsDouble(res.get(), val.get()));
		}

		return new Tensor(
			new Variavel[]{ res },
			1
		);
	}

	/**
	 * Retorna um {@code Tensor} contendo a soma dos elementos da 
     * instância local.
	 * @return {@code Tensor} resultado.
	 */
    public Tensor soma() {
		return reduce(0.0, (x, y) -> x + y);
    }

	/**
	 * Retorna um {@code Tensor} contendo a média aritmética dos 
     * elementos da instância local.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor media() {
        return soma().div(tam());
    }

	/**
	 * Retorna um {@code Tensor} contendo o valor máximo dentro dos 
     * elementos da instância local.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor max() {
		Variavel max = dados[0];

		final int n = tam();
		for (int i = 1; i < n; i++) {
			if (dados[i].maior(max)) max = dados[i];
		}

		return new Tensor(
			new double[]{ max.get() },
			1
		);
	}

	/**
	 * Retorna um {@code Tensor} contendo o valor mínimo dentro dos 
     * elementos da instância local.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor min() {
		Variavel min = dados[0];

		final int n = tam();
		for (int i = 1; i < n; i++) {
			if (dados[i].menor(min)) min = dados[i];
		}

		return new Tensor(
			new double[]{ min.get() },
			1
		);
	}

	/**
	 * Retorna um {@code Tensor} contendo o desvio padrão de acordo com os
     * elementos da instância local.
	 * @return {@code Tensor} resultado.
     */
	public Tensor desvp() {
		double media = media().item();
		double soma = 0.0d;

		final int n = tam();
		for (int i = 0; i < n; i++) {
			soma += Math.pow(dados[i].get() - media, 2);
		}

		return new Tensor(
			new double[]{ Math.sqrt(soma/n) },
			1
		);
	}

	/**
	 * Normaliza os valores do tensor dentro do intervalo especificado.
	 * @param min valor mínimo do intervalo.
	 * @param max valor máximo do intervalo.
	 * @return instância local alterada.
	 */
	public Tensor norm(Number min, Number max) {
		double valMin = min().item();
		double valMax = max().item();

		double intOrig = valMax - valMin;
		double intNovo = max.doubleValue() - min.doubleValue();

		return aplicar(
			x -> ((x - valMin) / intOrig) * intNovo + min.doubleValue()
		);
	}

	/**
	 * Aplica a função de ativação {@code ReLU} em todos os
	 * elementos do tensor.
	 * @return instância local alterada.
	 */
	public Tensor relu() {
		return aplicar(x -> x > 0.0 ? x : 0.0);
	}

	/**
	 * Aplica a função de ativação {@code Sigmoid} em todos os
	 * elementos do tensor.
	 * @return instância local alterada.
	 */
	public Tensor sigmoid() {
		return aplicar(x -> 1.0 / (1.0 + Math.exp(-x)));
	}

	/**
	 * Aplica a função de ativação {@code TanH} (Tangente Hiperbólica)
	 * em todos os elementos do tensor.
	 * @return instância local alterada.
	 */
	public Tensor tanh() {
		return aplicar(x -> 2.0 / (1.0 + Math.exp(-2 * x)) - 1.0);
	}

	/**
	 * Aplica a função de ativação {@code Atan} (Arco Tangente)
	 * em todos os elementos do tensor.
	 * @return instância local alterada.
	 */
	public Tensor atan() {
		return aplicar(x -> Math.atan(x));
	}

	/**
	 * Calcula o valor {@code seno} de todos os elementos do tensor.
	 * @return instância local alterada.
	 */
	public Tensor sin() {
		return aplicar(x -> Math.sin(x));
	}

	/**
	 * Calcula o valor {@code cosseno} de todos os elementos do tensor.
	 * @return instância local alterada.
	 */
	public Tensor cos() {
		return aplicar(x -> Math.cos(x));
	}

	/**
	 * Calcula o valor {@code tangente} de todos os elementos do tensor.
	 * @return instância local alterada.
	 */
	public Tensor tan() {
		return aplicar(x -> Math.tan(x));
	}

	/**
	 * Calcula o valor {@code absoluto} de cada elemento do do tensor.
	 * @return instância local alterada.
	 */
	public Tensor abs() {
		return aplicar(x -> Math.abs(x));
	}

	/**
	 * Calcula o valor {@code exponencial} de cada elemento do do tensor.
	 * @return instância local alterada.
	 */
	public Tensor exp() {
		return aplicar(x -> Math.exp(x));
	}

	/**
	 * Calcula o valor {@code logaritmo natural} de cada elemento do do tensor.
	 * @return instância local alterada.
	 */
	public Tensor log() {
		return aplicar(x -> Math.log(x));
	}

    /**
     * Retorna a quantidade de dimensões do tensor.
     * @return quantidade de dimensões do tensor.
     */
    public int numDim() {
        return shape.length;
    }

	/**
	 * Retorna um array contendo as dimensões do tensor.
	 * @return dimensões do tensor.
	 */
    public int[] shape() {
        return shape.clone();
    }

	/**
	 * Retorna uma String contendo as dimensões do tensor.
	 * @return dimensões do tensor em formato de String.
	 */
    public String shapeStr() {
        StringBuilder sb = new StringBuilder();

        sb.append("(");
		sb.append(shape[0]);
        for (int i = 1; i < shape.length; i++) {
            sb.append(", ").append(shape[i]);
        }
        sb.append(")");

        return sb.toString();
    }

	/**
	 * Compara todo o conteúdo da instância local, isso inclui as {@code dimensões}
	 * de cada tensor e seus {@code elementos individuais}.
	 * @param tensor {@code Tensor} desejado.
	 * @return {@code true} caso sejam iguais, {@code false} caso contrário.
	 */
	public boolean comparar(Tensor tensor) {
		if (!compararShape(tensor)) return false;

		final int n = tam();
		for (int i = 0; i < n; i++) {
			if (!dados[i].equals(tensor.dados[i])) return false;
		}

		return true;
	}

    /**
     * Verifica se o shape do tensor fornecido é igual ao shape
     * da instância local.
     * @param tensor {@code Tensor} desejado.
     * @return {@code true} caso as dimensões de ambos os tensores sejam
     * iguais, {@code false} caso contrário.
     */
    public boolean compararShape(Tensor tensor) {
        int n = shape.length;
        if (n != tensor.shape.length) return false;

        for (int i = 0; i < n; i++) {
            if (shape[i] != tensor.shape[i]) return false;
        }

        return true;
    }

	/**
	 * Retorna a quantidade total de elementos no tensor.
	 * @return número elementos do tensor.
	 */
	public int tam() {
		return dados.length;
	}

    /**
     * Calcula o tamanho em {@code bytes} do tensor, 
     * levando em consideração a arquitetura da JVM (32 ou 64 bits).
     * @return tamanho em bytes.
     */
	public long tamBytes() {
		String jvmBits = System.getProperty("sun.arch.data.model");
        long bits = Long.valueOf(jvmBits);

        long tamObj;
		// overhead da jvm
        if (bits == 32) tamObj = 8;
        else if (bits == 64) tamObj = 16;
        else throw new IllegalStateException(
            "\nSem suporte para plataforma de " + bits + " bits."
        );

		long tamVars = dados[0].tamanhoBytes() * tam();
		long tamShape = shape.length * 4; // int = 4 bytes
		return tamObj + tamVars + tamShape;
	}

	/**
	 * Retorna o conteúdo do tensor no formato de array
	 * <p>
	 * 		Por padrão as variáveis retornadas são as mesmas usadas pelo
	 * 		tensor, significa dizer que caso ela sofram alterações, isso
	 * 		é refletido automaticamente no tensor.
	 * </p>
	 * @return conteúdo do tensor.
	 */
	public Variavel[] paraArray() {
		return dados;
	}

	/**
	 * Retorna elementos específicos do conteúdo do tensor.
	 * <p>
	 * 		Por padrão as variáveis retornadas são as mesmas usadas pelo
	 * 		tensor, significa dizer que caso ela sofram alterações, isso
	 * 		é refletido automaticamente no tensor.
	 * </p>
	 * @param inicio índice de inicio (inclusivo).
	 * @param fim índice de fim (exclusivo).
	 * @return array de {@code Variaveis} do tensor de acordo com
	 * os índices.
	 */
	public Variavel[] paraArrayPorIndice(int inicio, int fim) {
		if (inicio < 0) {
			throw new IllegalArgumentException(
				"\nÍndice de inicio (" + inicio + ") inválido."
			);
		}
		if (fim > tam()) {
			throw new IllegalArgumentException(
				"\nÍndice de fim (" + fim + ") inválido."
			);
		}
		if (inicio >= fim) {
			throw new IllegalArgumentException(
				"\nÍndice de inicio (" + inicio + ") deve ser menor que índice de fim (" + fim + ")." 
			);
		}

		final int n = fim - inicio;
		Variavel[] arr = new Variavel[n];
		for (int i = 0; i < n; i++) {
			arr[i] = dados[i+inicio];// por padrão compartilhar variáveis.
		}

		return arr;
	}

	/**
	 * Retorna o conteúdo do tensor no formato de array {@code double[]}.
	 * @return conteúdo do tensor.
	 */
	public double[] paraArrayDouble() {
		final int n = tam();
		double[] arr = new double[n];
		for (int i = 0; i < n; i++) {
			arr[i] = dados[i].get();
		}

		return arr;
	}

	/**
	 * Configura o nome do tensor.
	 * @param nome novo nome.
	 * @return instância local alterada.
	 */
	public Tensor nome(String nome) {
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

    @Override
    public String toString() {
        return construirPrint();
    }

	/**
	 * Exibe, {@code via terminal}, todo o conteúdo do tensor.
	 */
	public void print() {
		System.out.println(construirPrint());
	}

	/**
	 * Monta as informações de exibição do tensor.
	 * @return string formatada.
	 */
    private String construirPrint() {
		final String identacao = " ".repeat(4);

		int maxCasasDecimais = 0;
		for (Variavel valor : dados) {
			String valorStr = ((Double)valor.get()).toString();
			int decimais = valorStr.length() - valorStr.indexOf('.') - 1;
			if (decimais > maxCasasDecimais) maxCasasDecimais = decimais;
		}

		int tamMaximo = -1;
        for (Variavel valor : dados) {
            String valorStr = valorStr(valor.get(), maxCasasDecimais);
			int tamValor = valorStr.length();
            if (tamValor > tamMaximo) tamMaximo = tamValor;
		}

        StringBuilder sb = new StringBuilder();

        int[] indices = new int[shape.length];
        boolean[] parentesisAbertos = new boolean[shape.length];

		sb.append(nome()).append(" ").append(shapeStr()).append(" = [").append("\n");

        sb.append(identacao);
        for (int n = 0; n < tam(); n++) {
            for (int i = 0; i < indices.length; i++) {
                if (!parentesisAbertos[i]) {
                    sb.append("[");
                    parentesisAbertos[i] = true;
                }
            }

            final String valorStr = valorStr(get(indices), maxCasasDecimais);
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

		// arrumar ultima linha antes do fim do print
		int n = identacao.length();
		if (numDim() > 1) n += 1;
        for (int i = 0; i < n; i++) {
            sb.deleteCharAt(sb.length()-1);
        }

		sb.append("]").append("\n");

        return sb.toString().trim();
	}

	/**
	 * Retorna o valor numérico formatado em string.
	 * @param x valor numérico. 
	 * @param casas quantidade de casas decimais desejada.
	 * @return valor formatado.
	 */
	private String valorStr(double x, int casas) {
		return String.format("%.0" + casas + "f", x).replace(',', '.');
	}

	/**
	 * Clona o conteúdo do tensor numa instância separada.
	 * @return clone da instância local.
	 */
    @Override
	public Tensor clone() {
		try {
			Tensor clone = (Tensor) super.clone();

			clone.shape = shape.clone();
			
			int n = tam();
			clone.dados = new Variavel[n];
			for (int i = 0; i < n; i++) {
				clone.dados[i] = new Variavel(dados[i]);
			}
			
			return clone;
		} catch (CloneNotSupportedException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public boolean equals(Object obj) {
		return (obj instanceof Tensor) && comparar((Tensor) obj);
	}

	@Override
	public Iterator<Variavel> iterator() {
		return new TensorIterator();
	}

	/**
	 * Iterador para usar com o tensor, usando para percorrer
	 * os elementos do tensor sequencialmente.
	 */
	class TensorIterator implements Iterator<Variavel> {

		/**
		 * Contador do índice atual.
		 */
		private int indice = 0;

		@Override
		public boolean hasNext() {
			return indice < tam();
		}

		@Override
		public Variavel next() {
			return dados[indice++];
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException(
				"\nSem suporte."
			);
		}
	}

	/**
	 * Exporta os dados do tensor em um arquivo externo.
	 * <h3>
	 *		Observação
	 * </h3>
	 * Para ler o tensor de um arquivo externo é necessário usar um
	 * {@code SerialTensor}, interface responsável pelo io de tensores.
	 * <p>
	 *		Exemplo:
	 * </p>
	 * <pre>
	 *import jnn.serializacao.SerialTensor;
	 *
	 *String caminho = ...
	 *SerialTensor st = new SerialTensor();
	 *Tensor t = st.ler(caminho);
	 * </pre>
	 * @param caminho caminho de destino.
	 */
	public void salvar(String caminho) {
		new SerialTensor().serializar(this, caminho);
	}

	/**
	 * Retorna um novo tensor contendo os dados da dimensão desejada.
	 * <p>
	 *		Exemplo:
	 * </p>
	 * <pre>
	 *tensor = [
	 *	[[1, 2, 3],
	 *	 [4, 5, 6]]
	 *]
	 * 
	 *subTensor0 = tensor.subTensor(0).
	 *subtensor0 = [
	 *	[1, 2, 3],
	 *]
	 * 
	 *subTensor1 = tensor.subTensor(1).
	 *subtensor1 = [
	 *	[4, 5, 6],
	 *]
	 * </pre>
	 * @param dim índice da subdimensão desejada.
	 * @return {@code Tensor} com o sub conteúdo.
	 */
	public Tensor subTensor(int dim) {
		if (tam() == 1) {
			throw new UnsupportedOperationException(
				"\nNão é possível obter um subtensor a partir de um tensor escalar."
			);
		}

		if (dim < 0 || dim >= shape[0]) {
			throw new IllegalArgumentException(
				"\nÍndice " + dim + " inválido."
			);
		}

		if (numDim() == 1) {// retornar tensor escalar
			return new Tensor(
				new Variavel[] { dados[dim] },
				1
			);
		}

		int[] novoShape = new int[shape.length-1];
		int i;
		for (i = 0; i < novoShape.length; i++) {
			novoShape[i] = shape[i+1];
		}

		int[] indices = new int[shape.length];
		indices[0] = dim;
		int inicio = indice(indices);

		for (i = 1; i < indices.length; i++) {
			indices[i] = shape[i]-1;
		}
		int fim = indice(indices);

		Variavel[] novosDados = new Variavel[calcularTamanho(novoShape)];
		for (i = inicio; i <= fim; i++) {
			novosDados[i-inicio] = dados[i];
		}

		return new Tensor(novosDados, novoShape);
	}

	/**
	 * Fatia o tensor de acordo com os índices fornecidos.
	 * <p>
	 *		Exemplo:
	 * </p>
	 * <pre>
	 *tensor = [
     *[[[ 1, 2],
     *  [ 3, 4]],
	 * 
     * [[ 5, 6],
     *  [ 7, 8]],
	 * 
     * [[ 9, 10],
     *  [11, 12]]]
	 *]
	 * 
	 *slice = tensor.slice(0, 2).
	 *slice = [
	 *	[[[1, 2],
	 *	  [3, 4]],
	 *
	 *	 [[5, 6],
	 *	  [7, 8]]]
	 *]
	 * </pre>
	 * @param inicio índice inicial (inclusivo).
	 * @param fim índice final (exclusivo).
	 * @return {@code Tensor} fatiado.
	 */
	public Tensor slice(int inicio, int fim) {
		if (tam() == 1) {
			throw new UnsupportedOperationException(
				"\nNão é possível obter um slice a partir de um tensor escalar."
			);
		}

		if (inicio < 0 || fim > shape[0] || inicio >= fim) {
			throw new IllegalArgumentException(
				"\nIntervalo inválido: [" + inicio + ", " + fim + "]"
			);
		}

		int[] novoShape = new int[shape.length];// manter shape original
		novoShape[0] = fim - inicio;

		for (int i = 1; i < shape.length; i++) {
			novoShape[i] = shape[i];
		}

		int[] idsInicio = new int[shape.length];
		int[] idsFim = new int[shape.length];
		idsInicio[0] = inicio;
		idsFim[0] = fim - 1;

		for (int i = 1; i < idsInicio.length; i++) {
			idsInicio[i] = 0;
			idsFim[i] = shape[i] - 1;
		}

		int idInicio = indice(idsInicio);
		int idFim = indice(idsFim);

		Variavel[] novosDados = new Variavel[calcularTamanho(novoShape)];
		for (int i = idInicio; i <= idFim; i++) {
			novosDados[i - idInicio] = dados[i];
		}

		return new Tensor(novosDados, novoShape);
	}

	/**
	 * Realiza uma operação a partir de dois {@code Tensor} que podem conter 
	 * shapes diferentes.
	 * <p>
	 *		Exemplo:
	 * </p>
	 * <pre>
	 *a = [
	 *	[[1, 2],
	 *	 [3, 4]]
	 *]
	 * 
	 * b = [2]
	 * 
	 * a.broadcast(b, (x, y) -> x + y);
	 * 
	 *res = [
	 *	[[3, 4],
	 *	 [5, 6]]
	 *]
	 * </pre>
	 * @param t {@code Tensor} para operação.
	 * @param op tipo de operação entre os tensores.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor broadcast(Tensor t, DoubleBinaryOperator op) {
		int[] outShape = broadcastShape(this.shape, t.shape);
		Tensor broadcast = new Tensor(outShape);

		int[] idBroad = new int[outShape.length];
		int total = broadcast.tam();

		for (int i = 0; i < total; i++) {
			unravelIndex(i, outShape, idBroad);

			int[] idsA = ajustarIndices(idBroad, this.shape);
			int[] idsB = ajustarIndices(idBroad, t.shape);

			double v = op.applyAsDouble(this.get(idsA), t.get(idsB));
			broadcast.set(v, idBroad);
		}

		return broadcast;
	}

	/**
	 * Determina o shape resultante do broadcasting
	 * @param shapeA {@code array} contendo o formato do {@code Tensor} A.
	 * @param shapeB {@code array} contendo o formato do {@code Tensor} B.
	 * @return {@code array} contendo o formato do {@code Tensor} resultante do broadcast.
	*/
	private int[] broadcastShape(int[] shapeA, int[] shapeB) {
		int n = Math.max(shapeA.length, shapeB.length);
		int[] broadShape = new int[n];

		for (int i = 0; i < n; i++) {
			int dimA = (shapeA.length - i - 1 >= 0) ? shapeA[shapeA.length - i - 1] : 1;
			int dimB = (shapeB.length - i - 1 >= 0) ? shapeB[shapeB.length - i - 1] : 1;

			if (dimA == dimB || dimA == 1 || dimB == 1) {
				broadShape[n - i - 1] = Math.max(dimA, dimB);
			} else {
				throw new IllegalArgumentException(
					"\nShapes incompatíveis para broadcasting: "
					+ Arrays.toString(shapeA) + " e " + Arrays.toString(shapeB)
				);
			}
		}

		return broadShape;
	}

	/**
	 * Converte índice linear em índices multidimensionais.
	 * @param idLinear índice linear dentro do {@code array} de dados do {@code Tensor}.
	 * @param shape formato do {@code Tensor}.
	 * @param out {@code array} de destino.
	*/
	private void unravelIndex(int idLinear, int[] shape, int[] out) {
		for (int i = shape.length - 1; i >= 0; i--) {
			out[i] = idLinear % shape[i];
			idLinear /= shape[i];
		}
	}

	/**
	 * Ajusta índices para lidar com broadcasting
	 * @param ids {@code array} de índices.
	 * @param shape {@code array} do formato do {@code Tensor}.
	 * @return índices corrigidos.
	*/
	private int[] ajustarIndices(int[] ids, int[] shape) {
		int[] out = new int[shape.length];
		int offset = ids.length - shape.length;

		for (int i = 0; i < shape.length; i++) {
			if (shape[i] == 1) {
				out[i] = 0;
			} else {
				out[i] = ids[i + offset];
			}
		}

		return out;
	}

}