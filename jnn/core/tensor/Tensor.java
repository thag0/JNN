package jnn.core.tensor;

import java.util.Iterator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.DoubleBinaryOperator;

//TODO broadcasting entre tensores

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
 *		Exemplo de criação:
 * </h2>
 * <pre>
 *Tensor tensor = new Tensor(2, 2);
 *tensor = [
 *  [[0.0, 0.0],
 *   [0.0, 0.0]]
 *]
 * </pre>
 * Algumas operações entre tensores são válidas desde que as dimensões
 * de ambos os tensores sejam iguais.
 * <pre>
 *Tensor a = new Tensor(2, 2);
 *a.preencer(1);
 *Tensor b = new Tensor(2, 2);
 *b.preencer(2);
 *a.add(b);//operação acontece dentro do tensor A
 *a = [
 *  [[3.0, 3.0],
 *   [3.0, 3.0]]
 *]
 * </pre>
 * @author Thiago Barroso, acadêmico de Engenharia da Computação pela
 * Universidade Federal do Pará, Campus Tucuruí. Maio/2024.
 */
public class Tensor implements Iterable<Variavel> {
    
	/**
	 * Dimensões do tensor.
	 */
	private int[] shape;

	/**
	 * Conjunto de elementos do tensor.
	 */
	private final Variavel[] dados;

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
		if (tensor == null) {
			throw new IllegalArgumentException(
				"O tensor fornecido é nulo."
			);
		}

        this.shape = tensor.shape.clone();

        int n = tensor.tamanho();
        dados = initDados(n);
		for (int i = 0; i < n; i++) {
			dados[i].set(tensor.dados[i]);
		}
    }

	/**
	 * Inicializa um tensor a partir de um array 4D primitivo.
	 * @param tensor tensor desejado.
	 */
	public Tensor(double[][][][] tensor) {
		if (tensor == null) {
			throw new IllegalArgumentException(
				"\nO tensor fornecido é nulo."
			);
		}

		this.shape = new int[]{
            tensor.length, 
            tensor[0].length, 
            tensor[0][0].length, 
            tensor[0][0][0].length
        };

		dados = initDados(shape[0] * shape[1] * shape[2] * shape[3]);
		copiar(tensor);
	}

	/**
	 * Inicializa um tensor a partir de um array 3D primitivo.
	 * @param tensor tensor desejado.
	 */
	public Tensor(double[][][] tensor) {
		if (tensor == null) {
			throw new IllegalArgumentException(
				"\nO tensor fornecido é nulo."
			);
		}

		this.shape = new int[]{
            tensor.length, 
            tensor[0].length, 
            tensor[0][0].length,
        };

		dados = initDados(shape[0] * shape[1] * shape[2]);
		copiar(tensor);
	}

	/**
	 * Inicializa um tensor a partir de um array 2D primitivo.
	 * @param mat matriz desejada.
	 */
	public Tensor(double[][] mat) {
		if (mat == null) {
			throw new IllegalArgumentException(
				"\nA matriz fornecida é nula."
			);
		}

		int col = mat[0].length;
		for (int i = 1; i < mat.length; i++) {
			if (mat[i].length != col) {
				throw new IllegalArgumentException(
					"\nA matriz deve conter a mesma quantidade de linhas para todas as colunas."
				);
			}
		}

		this.shape = copiarShape(new int[]{mat.length, mat[0].length});
		dados = initDados(mat.length * mat[0].length);
		copiar(mat);
	}

	/**
	 * Inicializa um tensor a partir de um array de variáveis.
	 * @param arr array desejado.
	 * @param dims dimensões desejadas.
	 */
    private Tensor(Variavel[] arr, int... dims) {
        shape = copiarShape(dims);

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
	 * Inicializar um tensor a partir de um conjunto de dados e formato
	 * pré-definidos.
	 * @param dados conjunto de dados desejado.
	 * @param shape formato do tensor.
	 */
	public Tensor(double[] dados, int... shape) {
		int[] s  = copiarShape(shape);
		int tam = calcularTamanho(s);
		if (tam != dados.length) {
			throw new IllegalArgumentException(
				"\nTamanho dos dados (" + dados.length + ") não corresponde ao " +
				"formato fornecido (" + tam + ")"
			);
		}
		this.shape = s;

		this.dados = initDados(tam);
		for (int i = 0; i < tam; i++) {
			this.dados[i].set(dados[i]);
		}
	}

    /**
     * Inicializa um novo tensor vazio a partir de um formato especificado.
     * @param shape formato desejado.
     */
    public Tensor(int... shape) {
        if (shape == null) {
            throw new IllegalArgumentException(
                "\nShape fornecido é nulo."
            );
        }

        int tam = calcularTamanho(shape);

        this.shape = copiarShape(shape);
        dados = initDados(tam);
    }

	/**
	 * Auxiliar na inicialização do conjunto de dados do tensor.
	 * @param tamanho tamanho desejado.
	 * @return array de dados alocado.
	 */
	private Variavel[] initDados(int tamanho) {
		Variavel[] d = new Variavel[tamanho];
		for (int i = 0; i < tamanho; i++) {
			d[i] = new Variavel(0.0d);
		}

		return d;
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
	private int[] copiarShape(int[] shape) {
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
	public Tensor reshape(int... dims) {
		int tamInicial = calcularTamanho(shape);

		int[] novoShape = copiarShape(dims);
		int novoTam = calcularTamanho(novoShape);

		if (tamInicial != novoTam) {
			throw new IllegalArgumentException(
				"\nQuatidade de elementos com as novas dimensões (" + novoTam +
				") deve ser igual a quantidade de elementos do tensor (" + tamanho() + ")."
			);
		}

		this.shape = novoShape;

		return this;
	}

	/**
	 * Cria uma nova visualização do tensor com as dimensões especificadas.
	 * @param dims dimensões desejadas.
	 * @return {@code Tensor} com a visualização desejada.
	 */
	public Tensor view(int... dims) {
		return new Tensor(dados, dims);
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

		int elementos = tamanho();

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
     * @param dims índices desejados.
     * @return índice correspondente no array de elementos do tensor.
     */
    private int indice(int... dims) {
        if (numDim() != dims.length) {
            throw new IllegalArgumentException(
				"\nNúmero de dimensões fornecidas " + dims.length + 
				" não corresponde às " + numDim() + " do tensor."
			);
        }
    
        int id = 0;
        int multiplicador = 1;
    
        for (int i = shape.length - 1; i >= 0; i--) {
            if (dims[i] < 0 || dims[i] >= shape[i]) {
                throw new IllegalArgumentException(
					"\nÍndice " + dims[i] + " fora dos limites para a dimensão " + i +
					" (tamanho = " + shape[i] + ");"
				);
            }
            id += dims[i] * multiplicador;
            multiplicador *= shape[i];
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
    public void set(double x, int... ids) {
        dados[indice(ids)].set(x);
    }

	/**
	 * Edita o valor do tensor usando uma variável.
	 * @param var variável com valor desejado.
	 * @param ids índices para atribuição.
	 */
	public void set(Variavel var, int... ids) {
		dados[indice(ids)].set(var);
	}

	/**
	 * Preenche todo o conteúdo do tensor com o valor fornecido.
	 * @param valor valor desejado.
	 * @return instância local alterada.
	 */
	public Tensor preencher(double valor) {
		final int n = tamanho();
		for (int i = 0; i < n; i++) {
			dados[i].set(valor);
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
		int tam = tamanho();

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
	public Tensor zerar() {
        final int n = tamanho();
		for (int i = 0; i < n; i++) {
			dados[i].set(0.0);
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

		int n = tamanho();
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

		if ((tamanho() != arr.length)) {
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
		if (tamanho() != tensor.tamanho()) {
			throw new IllegalArgumentException(
				"\nOs tensores devem conter o mesmo número de elementos. Local = " + tamanho() + 
				"e recebido = " + tensor.tamanho()
			);
		}

		int n = tamanho();
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

		final int n = tamanho();
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

		for (int i = 0; i < elementos.length; i++) {
			dados[i].set(elementos[i]);
		}
		
		return this;
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

        int n = tamanho();
        for (int i = 0; i < n; i++) {
            dados[i].add(tensor.dados[i]);
        }

        return this;
    }

	/**
	 * Adiciona o valor informado ao conteúdo do tensor.
	 * @param valor valor desejado.
	 * @param ids índices desejados para adição.
	 * @return instância local alterada.
	 */
	public Tensor add(double valor, int... ids) {
		dados[indice(ids)].add(valor);
		return this;
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

        int n = tamanho();
        for (int i = 0; i < n; i++) {
            dados[i].sub(tensor.dados[i]);
        }

        return this;
    }

	/**
	 * Subtrai o valor informado ao conteúdo do tensor.
	 * @param valor valor desejado.
	 * @param ids índices desejados para adição.
	 * @return instância local alterada.
	 */
	public Tensor sub(double valor, int... ids) {
		dados[indice(ids)].sub(valor);
		return this;
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
    public Tensor mult(Tensor tensor) {
        if (!compararShape(tensor)) {
            throw new IllegalArgumentException(
                "\nTensor fornecido deve conter o mesmo shape."
            );
        }

        int n = tamanho();
        for (int i = 0; i < n; i++) {
            dados[i].mult(tensor.dados[i]);
        }

        return this;
    }

	/**
	 * Multiplica o valor informado ao conteúdo do tensor.
	 * @param valor valor desejado.
	 * @param ids índices desejados para adição.
	 * @return instância local alterada.
	 */
	public Tensor mult(double valor, int... ids) {
		dados[indice(ids)].mult(valor);
		return this;
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

        int n = tamanho();
        for (int i = 0; i < n; i++) {
            dados[i].div(tensor.dados[i]);
        }

        return this;
    }

	/**
	 * Divide o valor contido no tensor pelo valor informado.
	 * @param valor valor desejado.
	 * @param ids índices desejados para adição.
	 * @return instância local alterada.
	 */
	public Tensor div(double valor, int... ids) {
		dados[indice(ids)].div(valor);
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

		if (numDim() == 1) return this; // não fazer nada com tensores escalares
	
		if (shape[dim] != 1) {
			return this; // não alterar dimensões com tamanho != 1
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
	
		final int tam = tamanho();
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
		if (fun == null) {
			throw new IllegalArgumentException(
				"\nFunção recebida é nula."
			);
		}

		for (int i = 0; i < dados.length; i++) {
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
		if (tensor == null) {
			throw new IllegalArgumentException(
				"\nTensor fornecido é nulo."
			);
		}
		if (!compararShape(tensor)) {
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
			dados[i].set(fun.applyAsDouble(tensor.dados[i].get()));
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
		if (a == null || b == null) {
			throw new IllegalArgumentException(
				"\nOs tesores fornecidos não podem ser nulos."
			);
		}
		if (!compararShape(a)) {
			throw new IllegalArgumentException(
				"\nAs dimensões do tensor A " + a.shapeStr() +
				" e as da instância local " + shapeStr() + " devem ser iguais."
			);
		}
		if (!compararShape(b)) {
			throw new IllegalArgumentException(
				"\nAs dimensões do tensor B " + b.shapeStr() +
				" e as da instância local " + shapeStr() + " devem ser iguais."
			);
		}
		if (fun == null) {
			throw new IllegalArgumentException(
				"\nFunção recebida é nula."
			);
		}

		for (int i = 0; i < dados.length; i++) {
			dados[i].set(fun.applyAsDouble(a.dados[i].get(), b.dados[i].get()));
		}

		return this;
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
		if (fun == null) {
			throw new IllegalArgumentException(
				"\nFunção recebida é nula."
			);
		}

		Tensor t = new Tensor(shape());

		for (int i = 0; i < t.tamanho(); i++) {
			t.dados[i].set(fun.applyAsDouble(dados[i].get()));
		}

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
		if (fun == null) {
			throw new IllegalArgumentException(
				"\nFunção recebida é nula."
			);
		}

		if (!compararShape(tensor)) {
			throw new IllegalArgumentException(
				"\nTensor " + tensor.shapeStr() + " deve conter mesmo formato do " +
				"tensor local " + shapeStr()
			);
		}

		Tensor t = new Tensor(shape());

		for (int i = 0; i < t.tamanho(); i++) {
			t.dados[i].set(fun.applyAsDouble(dados[i].get(), tensor.dados[i].get()));
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
	public Tensor reduce(double in, DoubleBinaryOperator fun) {
		if (fun == null) {
			throw new IllegalArgumentException(
				"\nFunção de redução não pode ser nula."
			);
		}

		Variavel res = new Variavel(in);
		for (Variavel val : dados) {
			res.set(fun.applyAsDouble(res.get(), val.get()));
		}

		return new Tensor(new Variavel[]{ res }, 1);
	}

	/**
	 * Retorna um {@code Tensor} contendo a soma dos elementos da 
     * instância local.
	 * @return {@code Tensor} resultado.
	 */
    public Tensor soma() {
        double soma = 0.0d;
        final int n = tamanho();
        for (int i = 0; i < n; i++) {
            soma += dados[i].get();
        }

        return new Tensor(new Variavel[]{ new Variavel(soma) }, 1);
    }

	/**
	 * Retorna um {@code Tensor} contendo a média aritmética dos 
     * elementos da instância local.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor media() {
        double media = soma().item() / tamanho();
        return new Tensor(new Variavel[]{ new Variavel(media) }, 1);
    }

	/**
	 * Retorna um {@code Tensor} contendo o valor máximo dentro dos 
     * elementos da instância local.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor maximo() {
		double max = dados[0].get();
		final int tam = tamanho();

		for (int i = 1; i < tam; i++) {
			if (dados[i].get() > max) max = dados[i].get();
		}

		return new Tensor(new Variavel[]{ new Variavel(max) }, 1);
	}

	/**
	 * Retorna um {@code Tensor} contendo o valor mínimo dentro dos 
     * elementos da instância local.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor minimo() {
		double min = dados[0].get();
		final int tam = tamanho();

		for (int i = 1; i < tam; i++) {
			if (dados[i].get() < min) min = dados[i].get();
		}

		return new Tensor(new Variavel[]{ new Variavel(min) }, 1);
	}

	/**
	 * Retorna um {@code Tensor} contendo o desvio padrão de acordo com os
     * elementos da instância local.
	 * @return {@code Tensor} resultado.
     */
	public Tensor desvp() {
		double media = media().item();
		double soma = 0.0d;
        final int n = tamanho();

		for (int i = 0; i < n; i++) {
			soma += Math.pow(dados[i].get() - media, 2);
		}

		return new Tensor(new Variavel[]{ 
			new Variavel(Math.sqrt(soma / tamanho()))
		}, 1);
	}

	/**
	 * Normaliza os valores do tensor dentro do intervalo especificado.
	 * @param min valor mínimo do intervalo.
	 * @param max valor máximo do intervalo.
	 * @return instância local alterada.
	 */
	public Tensor normalizar(double min, double max) {
		double valMin = minimo().item();
		double valMax = maximo().item();

		double intOriginal = valMax - valMin;
		double intNovo = max - min;

		aplicar(x -> {
			return ((x - valMin) / intOriginal) * intNovo + min;
		});

		return this;
	}

	/**
	 * Aplica a função de ativação {@code ReLU} em todos os
	 * elementos do tensor.
	 * @return instância local alterada.
	 */
	public Tensor relu() {
		return aplicar(x -> x > 0 ? x : 0);
	}

	/**
	 * Aplica a função de ativação {@code Sigmoid} em todos os
	 * elementos do tensor.
	 * @return instância local alterada.
	 */
	public Tensor sigmoid() {
		return aplicar(x -> 1 / (1 + Math.exp(-x)));
	}

	/**
	 * Aplica a função de ativação {@code TanH} (Tangente Hiperbólica)
	 * em todos os elementos do tensor.
	 * @return instância local alterada.
	 */
	public Tensor tanh() {
		return aplicar(x -> 2 / (1 + Math.exp(-2 * x)) - 1);
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
        return shape;
    }

	/**
	 * Retorna uma String contendo as dimensões do tensor.
	 * @return dimensões do tensor em formato de String.
	 */
    public String shapeStr() {
        StringBuilder sb = new StringBuilder();

        sb.append("(");
        for (int i = 0; i < shape.length; i++) {
            sb.append(shape[i]).append(", ");
        }
        sb.deleteCharAt(sb.length()-1);
        sb.deleteCharAt(sb.length()-1);
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

		for (int i = 0; i < dados.length; i++) {
			if (dados[i].get() != tensor.dados[i].get()) return false;
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
	public int tamanho() {
		return dados.length;
	}

    /**
     * Calcula o tamanho em {@code bytes} do tensor, 
     * levando em consideração a arquitetura da JVM (32 ou 64 bits).
     * @return tamanho em bytes.
     */
	public long tamanhoBytes() {
		String jvmBits = System.getProperty("sun.arch.data.model");
        long bits = Long.valueOf(jvmBits);

        long tamObj;
		// overhead da jvm
        if (bits == 32) tamObj = 8;
        else if (bits == 64) tamObj = 16;
        else throw new IllegalStateException(
            "\nSem suporte para plataforma de " + bits + " bits."
        );

		long tamVars = dados[0].tamanhoBytes() * tamanho();
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
		if (fim > tamanho()) {
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
		double[] arr = new double[dados.length];
		for (int i = 0; i < arr.length; i++) {
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
        for (int n = 0; n < tamanho(); n++) {
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
		return new Tensor(this);
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
			return indice < tamanho();
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

}