package jnn.core.tensor;

import java.util.Iterator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.DoubleBinaryOperator;

/**
 * <h2>
 *		Tensor multidimensional
 * </h2>
 * Implementação de um array multidimensional com finalidade de simplificar 
 * o uso de estrutura de dados dentro da biblioteca.
 * <p>
 * 		O tensor possui algumas funções próprias com intuito de aproveitar a
 * 		velocidade de processamento usando um único array contendo os dados do dele.
 * </p>
 * <h2>
 *		Exemplo de criação:
 * </h2>
 * <pre>
 *Tensor tensor = new Tensor(1, 1, 2, 2);
 *Tensor tensor = new Tensor(new int[]{2, 2});
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
 * 
 * @author Thiago Barroso, acadêmico de Engenharia da Computação pela
 * Universidade Federal do Pará, Campus Tucuruí. Maio/2024.
 */
public class Tensor implements Iterable<Double> {
    
	/**
	 * Dimensões do tensor.
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
        this.dados = new double[n];
        System.arraycopy(tensor.dados, 0, this.dados, 0, n);
    }

	/**
	 * Inicializa um tensor a partir de um array quadridimensional primitivo.
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

		this.dados = new double[shape[0] * shape[1] * shape[2] * shape[3]];

		copiar(tensor);
	}

	/**
	 * Inicializa um tensor a partir de um array tridimensional primitivo.
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

		this.dados = new double[shape[0] * shape[1] * shape[2]];

		copiar(tensor);
	}

	/**
	 * Inicializa um tensor a partir de um array bidimensional primitivo.
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

		this.shape = new int[]{mat.length, mat[0].length};
		this.dados = new double[shape[0] * shape[1]];

		copiar(mat);
	}

	/**
	 * Inicializa um tensor a partir de um array primitivo.
	 * @param arr array desejado.
	 */
    public Tensor(double[] arr) {
        shape = new int[]{arr.length};
        dados = new double[arr.length];
		System.arraycopy(arr, 0, dados, 0, arr.length);
    }

    /**
     * Inicializa um novo tensor a partir de um formato especificado.
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
        dados = new double[tam];
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
        int inicio = 0;
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] != 1) {
                inicio = i;
                break;
            }
        }

        int[] s = new int[shape.length-inicio];
        System.arraycopy(shape, inicio, s, 0, s.length);

        return s;
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
	 * @param dim array contendo as novas dimensões (dim1, dim2, dim3, dim4).
	 * @return instância local alterada.
	 */
	public Tensor reshape(int... dims) {
		int tamInicial = calcularTamanho(shape);

		int[] dimsUteis = copiarShape(dims);
		int tamDesejado = calcularTamanho(dimsUteis);

		if (tamInicial != tamDesejado) {
			throw new IllegalArgumentException(
				"\nA quatidade de elementos com as novas dimensões (" + tamDesejado +
				") deve ser igual a quantidade de elementos do tensor (" + tamanho() + ")."
			);
		}

		Tensor novo = new Tensor(dimsUteis);
		novo.copiarElementos(dados);

		return novo;
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
					"\nÍndice " + dims[i] + " fora dos limites para a dimensão " + i
				);
            }
            id += dims[i] * multiplicador;
            multiplicador *= shape[i];
        }
    
        return id;
    }

	/**
	 * Retorna o elemento do tensor de acordo com os índices fornecidos.
	 * @param indices índices desejados para busca.
	 * @return valor de acordo com os índices.
	 */
    public double get(int... ids) {
        return dados[indice(ids)];
    }

	/**
	 * Edita o valor do tensor usando o valor informado.
	 * @param indices índices para atribuição.
	 * @param valor valor desejado.
	 */
    public void set(double valor, int... ids) {
        dados[indice(ids)] = valor;
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
	public Tensor zerar() {
        final int n = tamanho();
		for (int i = 0; i < n; i++) {
			dados[i] = 0.0d;
		}

		return this;
	}

	/**
	 * Copia todo o conteúdo do tensor na instância local.
	 * @param tensor tensor desejado.
	 * @return instância local alterada.
	 */
	public Tensor copiar(Tensor tensor) {
		if (!compararShape(tensor)) {
			throw new IllegalArgumentException(
				"\nDimensões " + shapeStr() + " incompatíveis com as do" +
				" tensor recebido " + tensor.shapeStr()
			);
		}

		System.arraycopy(tensor.dados, 0, this.dados, 0, tamanho());

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
						this.dados[cont++] = arr[i][j][k][l];
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
					this.dados[cont++] = arr[i][j][k];
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
        if (numDim() != 2) {
            throw new IllegalArgumentException(
                "\nTensor tem " + numDim() + " dimensões, mas deve" +
                " ter 2."
            );
        }

        int d1 = shape[0];
        int d2 = shape[1];

		if ((d1 != arr.length) ||
			(d2 != arr[0].length)) {
			throw new IllegalArgumentException(
				"\nDimensões do tensor " + shapeStr() +
				" incompatíveis com as do array recebido ("
				+ arr.length + ", " + arr[0].length + ")."
			);
		}

		int cont = 0;
		for (int i = 0; i < d1; i++) {
			System.arraycopy(arr[i], 0, dados, cont, arr[i].length);
			cont += arr[i].length;
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

		System.arraycopy(arr, 0, dados, 0, tamanho());

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

		System.arraycopy(elementos, 0, dados, 0, tamanho());

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
                "\nTensor fornecido deve conter o mesmo shape."
            );
        }

        int n = tamanho();
        for (int i = 0; i < n; i++) {
            dados[i] += tensor.dados[i];
        }

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
            dados[i] -= tensor.dados[i];
        }

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
            dados[i] *= tensor.dados[i];
        }

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
            dados[i] /= tensor.dados[i];
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
			dados[i] = fun.applyAsDouble(dados[i]);
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
	public Tensor map(DoubleUnaryOperator fun) {
		if (fun == null) {
			throw new IllegalArgumentException(
				"\nFunção recebida é nula."
			);
		}

		Tensor t = new Tensor(shape());

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
	public Tensor reduce(double in, DoubleBinaryOperator fun) {
		if (fun == null) {
			throw new IllegalArgumentException(
				"\nFunção de redução não pode ser nula."
			);
		}

		double res = in;
		for (double val : dados) {
			res = fun.applyAsDouble(res, val);
		}

		return new Tensor(new double[]{ res });
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
            soma += dados[i];
        }

        return new Tensor(new double[]{ soma });
    }

	/**
	 * Retorna um {@code Tensor} contendo a média aritmética dos 
     * elementos da instância local.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor media() {
        double media = soma().item() / tamanho();
        return new Tensor(new double[]{ media });
    }

	/**
	 * Retorna um {@code Tensor} contendo o valor máximo dentro dos 
     * elementos da instância local.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor maximo() {
		double max = dados[0];
		final int tam = tamanho();

		for (int i = 1; i < tam; i++) {
			if (dados[i] > max) max = dados[i];
		}

		return new Tensor(new double[] { max });
	}

	/**
	 * Retorna um {@code Tensor} contendo o valor mínimo dentro dos 
     * elementos da instância local.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor minimo() {
		double min = dados[0];
		final int tam = tamanho();

		for (int i = 1; i < tam; i++) {
			if (dados[i] < min) min = dados[i];
		}

		return new Tensor(new double[] { min });
	}

	/**
	 * Retorna um {@code Tensor} contendo o desvio padrão de acordo com os
     * elementos da instância local.
	 * @return {@code Tensor} resultado.
     */
	public double desvp() {
		double media = media().item();
		double soma = 0.0d;
        final int n = tamanho();

		for (int i = 0; i < n; i++) {
			soma += Math.pow(dados[i] - media, 2);
		}

		return Math.sqrt(soma / tamanho());
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

        final int n = tamanho();
		for (int i = 0; i < n; i++) {
			dados[i] = ((dados[i] - valMin) / intOriginal) * intNovo + min;
		}

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
			if (dados[i] != tensor.dados[i]) return false;
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
	 * Retorna o conteúdo do tensor no formato de array
	 * @return conteúdo do tensor.
	 */
	public double[] paraArray() {
		return dados.clone();
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

		sb.append(nome()).append(" ").append(shapeStr()).append(" = [").append("\n");

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
	 * Clona o conteúdo do tensor numa instância separada.
	 * @return clone da instância local.
	 */
    @Override
	public Tensor clone() {
		return new Tensor(this);
	}

	@Override
	public boolean equals(Object obj) {
		return (obj instanceof Tensor4D) && comparar((Tensor) obj);
	}

	@Override
	public Iterator<Double> iterator() {
		return new TensorIterator();
	}

	/**
	 * Iterador para usar com o tensor, usando para percorrer
	 * os elementos do tensor sequencialmente.
	 */
	class TensorIterator implements Iterator<Double> {

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
