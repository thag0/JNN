package jnn.core.tensor;

import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;

import jnn.core.tensor.operadores.FloatBinaryOperator;
import jnn.core.tensor.operadores.FloatTernaryOperator;
import jnn.core.tensor.operadores.FloatUnaryOperator;
import jnn.io.seriais.SerialTensor;

/**
 * <h2>
 *		Tensor Multidimensional
 * </h2>
 *		Um {@code Tensor} representa um vetor que pode conter diversas dimensões,
 *		cada uma contendo um tamanho fixo de elementos.
 * <p>
 *		Esta implementação de tensor considera o formato {@code row-major} para
 *		armazenamento em memória e manipulação.
 * </p>
 * <p>
 *		A maioria das implementações considera alterações {@code in-place} sendo
 *		evitado ao máximo alocação de memória para reduzir pressão no Garbage Collector.
 * </p>
 * @author Thiago Barroso, acadêmico de Engenharia da Computação pela
 * Universidade Federal do Pará, Campus Tucuruí. Maio/2024.
 * @see <a href="https://github.com/thag0"> Github </a>
 */
public class Tensor implements Iterable<Float>, Cloneable {
    
	/**
	 * Dimensões do tensor.
	 */
	private int[] shape;

	/**
	 * Strides do tensor. Cada posição indica o salto no array de dados
	 * necessário para avançar uma unidade na respectiva dimensão.
	 */
	private int[] stride;

	/**
	 * Conjunto de dados do tensor.
	 */
	private TensorData data;

	/**
	 * Nome do tensor.
	 */
	private String name = getClass().getSimpleName();

	/**
	 * Inicializa um tensor a partir de outra instância.
	 * <p>
	 *		O conteúdo do tensor recebido será copiado.
	 * </p>
	 * @param t {@code Tensor} base.
	 */
    public Tensor(Tensor t) {
        this.shape = initShape(t.shape());
        int n = t.len();
        data = initData(n);
		copyElements(t);
		this.stride = initStrides(shape);
    }

	/**
	 * Inicializa um tensor a partir de um array 5D primitivo.
	 * @param arr {@code array} base.
	 */
	public Tensor(float[][][][][] arr) {
		if (arr == null) {
			throw new IllegalArgumentException(
				"\nArray nulo."
			);
		}

		shape = initShape(
            arr.length, 
            arr[0].length, 
            arr[0][0].length, 
            arr[0][0][0].length,
            arr[0][0][0][0].length
        );

		data = initData(shape[0] * shape[1] * shape[2] * shape[3] * shape[4]);
		copy(arr);
		this.stride = initStrides(shape);
	}

	/**
	 * Inicializa um tensor a partir de um array 4D primitivo.
	 * @param arr {@code array} base.
	 */
	public Tensor(float[][][][] arr) {
		if (arr == null) {
			throw new IllegalArgumentException(
				"\nArray nulo."
			);
		}

		this.shape = new int[]{
            arr.length, 
            arr[0].length, 
            arr[0][0].length, 
            arr[0][0][0].length
        };

		data = initData(shape[0] * shape[1] * shape[2] * shape[3]);
		copy(arr);
		this.stride = initStrides(shape);
	}

	/**
	 * Inicializa um tensor a partir de um array 3D primitivo.
	 * @param arr {@code array} base.
	 */
	public Tensor(float[][][] arr) {
		if (arr == null) {
			throw new IllegalArgumentException(
				"\nArray nulo."
			);
		}

		this.shape = new int[]{
            arr.length, 
            arr[0].length, 
            arr[0][0].length,
        };

		data = initData(shape[0] * shape[1] * shape[2]);
		copy(arr);
		this.stride = initStrides(shape);
	}

	/**
	 * Inicializa um tensor a partir de um array 2D primitivo.
	 * @param arr {@code array} base.
	 */
	public Tensor(float[][] arr) {
		if (arr == null) {
			throw new IllegalArgumentException(
				"\nArray nulo."
			);
		}

		int col = arr[0].length;
		for (int i = 1; i < arr.length; i++) {
			if (arr[i].length != col) {
				throw new IllegalArgumentException(
					"\nO array deve conter a mesma quantidade de linhas para todas as colunas."
				);
			}
		}

		this.shape = initShape(new int[]{arr.length, arr[0].length});
		data = initData(arr.length * arr[0].length);
		copy(arr);
		this.stride = initStrides(shape);
	}

	/**
	 * Inicializa um tensor a partir de um array primitivo.
	 * @param arr {@code array} base.
	 */
	public Tensor(float[] arr) {
		int[] shape = {arr.length};
		int[] s  = initShape(shape);
		int tam = calcLen(s);
		if (tam != arr.length) {
			throw new IllegalArgumentException(
				"\nTamanho dos dados (" + arr.length + ") não corresponde ao " +
				"formato fornecido (" + tam + ")"
			);
		}
		
		this.shape = s;
		this.stride = initStrides(shape);
		this.data = new TensorData(arr);
	}

    /**
     * Inicializa um novo tensor {@code vazio} a partir de um formato especificado.
     * @param shape formato desejado.
     */
    public Tensor(int... shape) {
        if (shape == null) {
            throw new IllegalArgumentException(
                "\nShape nulo."
            );
        }

        int tam = calcLen(shape);
        this.shape = initShape(shape);
        data = initData(tam);
		this.stride = initStrides(shape);
    }

	/**
	 * Inicializa um tensor a partir de um conjunto de dados.
	 * @param data dados base..
	 * @param shape dimensões desejadas.
	 * @param stride strides do novo {@code Tensor}.
	 */
	private Tensor(TensorData data, int[] shape, int[] stride) {
		if (data == null) {
			throw new IllegalArgumentException("\nConjunto de dados nulo.");
		}
		if (shape == null || stride == null) {
			throw new IllegalArgumentException("\nShape e strides não podem ser nulos.");
		}
		if (shape.length != stride.length) {
			throw new IllegalArgumentException("\nShape e strides devem ter o mesmo comprimento.");
		}

		this.data = data;
		this.shape  = Arrays.copyOf(shape,   shape.length);
		this.stride = Arrays.copyOf(stride, stride.length);
	}

	/**
	 * Auxiliar na inicialização do conjunto de dados do tensor.
	 * @param n tamanho do conjunto de elementos.
	 * @return {@code TensorData} contendo os elementos alocados.
	 */
	private TensorData initData(int n) {
		return new TensorData(n);
	}

    /**
     * Copia valores relevantes para o formato do tensor.
     * @param shape shape desejado.
     * @return shape com valores úteis.
     */
	private int[] initShape(int... shape) {
		if (shape.length == 0) {
			throw new IllegalArgumentException(
				"\nShape vazio."
			);
		}

		return shape.clone();
	}

	/**
	 * Calcula os strides do tensor a partir do shape. O último stride é 1
	 * (avanço unitário no array), e os anteriores são obtidos pelo produto
	 * acumulado dos tamanhos das dimensões seguintes.
	 * @param shape formato do {@code Tensor} desejado.
	 * @return strides calculados.
	 */
	private int[] initStrides(int[] shape) {
		int[] stride = new int[shape.length];
		
		stride[shape.length - 1] = 1;
		for (int i = shape.length - 2; i >= 0; i--) {
			stride[i] = stride[i + 1] * shape[i + 1];
		}

		return stride;
	}

	/**
	 * Calcula a quantidade de elementos de acordo com o formato informado.
	 * @param shape formato base.
	 * @return quantidade de elementos a partir do shape dado.
	 */
    private int calcLen(int[] shape) {
        if (shape.length == 0) return 0;

        int tam = 1;
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] < 1) {
                throw new IllegalArgumentException(
                    "\nShape informado deve conter valores maiores que 0."
                );
            }

            tam *= shape[i];
        }

        return tam;
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
	 * @param shape array contendo as novas dimensões.
	 * @return {@code view} do {@code Tensor}.
	 */
	public Tensor reshape(int... shape) {
		int[] novoShape = initShape(shape);
		int novoTam = calcLen(novoShape);

		if (len() != novoTam) {
			throw new IllegalArgumentException(
				"\nQuatidade de elementos com as novas dimensões (" + novoTam +
				") deve ser igual a quantidade de elementos do tensor (" + len() + ")."
			);
		}

		return new Tensor(
			data,
			novoShape,
			initStrides(novoShape)
		);
	}

	/**
	 * Transpõe o conteúdo do tensor.
	 * <p>
	 * 		Essa função tem como comportamento padrão trocar apenas os últimos dois eixos do tensor.
	 * </p>
	 * Para uma transposição mais completa, use {@code permutar()}
	 * @return uma {@code view} do {@code Tensor} transposto.
	 */
    public Tensor transpose() {
		int ndim = numDim();
		
		if (ndim < 2) {
			throw new UnsupportedOperationException(
				"\nTensor deve conter pelo menos duas dimensões."
			);
		}

		int[] novoShape = shape.clone();
		int[] novoStrides = stride.clone();

		int ultimo = ndim - 1;
		int penultimo = ndim - 2;

		int tmpShape = novoShape[penultimo];
		novoShape[penultimo] = novoShape[ultimo];
		novoShape[ultimo] = tmpShape;

		int tmpStride = novoStrides[penultimo];
		novoStrides[penultimo] = novoStrides[ultimo];
		novoStrides[ultimo] = tmpStride;

		return new Tensor(data, novoShape, novoStrides);
	}

	/**
	 * Retorna um {@code Tensor} com os eixos permutados.
	 * <p>
	 * 		Exemplo:
	 * </p>
	 * Tensor t = new Tensor(2, 3, 1)// shape (2, 3, 1)
	 * Tensor p = t.permutar(2, 0, 1)// shape (1, 2, 3)
	 * @param axis nova ordem de eixos.
	 * @return uma {@code view} do {@code Tensor} permutado.
	 */
	public Tensor permute(int... axis) {
		if (axis.length != shape.length) {
			throw new IllegalArgumentException(
				"Número de eixos no permute (" + axis.length +
				") diferente da dimensão do tensor (" + shape.length + ")"
			);
		}

		boolean[] eixoVisto = new boolean[shape.length];
		for (int eixo : axis) {
			if (eixo < 0 || eixo >= shape.length) {
				throw new IllegalArgumentException("Eixo inválido: " + eixo);
			}
			if (eixoVisto[eixo]) {
				throw new IllegalArgumentException("Eixo repetido: " + eixo);
			}
			eixoVisto[eixo] = true;
		}

		int[] novoShape = new int[shape.length];
		int[] novoStrides = new int[shape.length];

		for (int i = 0; i < axis.length; i++) {
			novoShape[i] = shape[axis[i]];
			novoStrides[i] = stride[axis[i]];
		}

		return new Tensor(data, novoShape, novoStrides);
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
	public Tensor tile(int n) {
		int[] shapeAntigo = shape();
		int tamAntigo = len();
		float[] dt = data.data();

		int[] shapeNovo = new int[shapeAntigo.length + 1];
		shapeNovo[0] = n;
		System.arraycopy(shapeAntigo, 0, shapeNovo, 1, shapeAntigo.length);

		Tensor bloco = new Tensor(shapeNovo);
		float[] db = bloco.array();

		for (int i = 0; i < n; i++) {
			int offset = i * tamAntigo;
			System.arraycopy(dt, 0, db, offset, tamAntigo);
		}

		return bloco;
	}

    /**
     * Calcula o índice de um elementos dentro do conjunto de dados do tensor.
     * @param ids índices desejados.
     * @return índice correspondente no array de elementos do tensor.
     */
    private int index(int... ids) {
		if (stride.length != ids.length) {
			throw new IllegalArgumentException(
				"Número de dimensões fornecidas " + ids.length +
				" não corresponde às " + stride.length + " do tensor."
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
			id += ids[i] * stride[i];
		}
		return id;
    }

	/**
	 * Retorna o elemento do tensor de acordo com os índices fornecidos.
	 * @param i índices desejados para busca.
	 * @return valor de acordo com os índices.
	 */
    public float get(int... i) {
        return data.get(index(i));
    }

	/**
	 * Edita o valor do tensor usando o valor informado.
	 * @param x valor desejado.
	 * @param i índices para atribuição.
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor set(Number x, int... i) {
        data.set(x.floatValue(), index(i));
		return this;
	}

	/**
	 * Preenche todo o conteúdo do tensor com o valor fornecido.
	 * @param x valor desejado.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor fill(Number x) {
		data.fill(x.floatValue());
		return this;
	}

	/**
	 * Preenche o conteúdo do tensor usando um contador iniciado com
	 * valor 1 que é alterado a cada elemento.
	 * @param pos contador crescente (1, 2, 3, ...), caso falso o
	 * contador é decrescente (-1, -2, -3, ...).
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor fillCount(boolean pos) {
		data.fillCount(pos);
		return this;
	}

	/**
	 * Zera todo o conteúdo o tensor.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor zero() {
		data.zero();
		return this;
	}

	/**
	 * Copia todo o conteúdo do tensor na instância local.
	 * @param t {@code Tensor} desejado.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor copy(Tensor t) {
		if (!compareShape(t)) {
			throw new IllegalArgumentException(
				"\nDimensões locais " + shapeStr() + " incompatíveis com as do" +
				" tensor recebido " + t.shapeStr()
			);
		}

		data.copy(t.data);

		return this;
	}

	/**
	 * Copia todo o conteúdo do array na instância local.
	 * @param arr {@code array} base.
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor copy(float[][][][][] arr) {
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
		float[] d = array();
		for (int i = 0; i < d1; i++) {
			for (int j = 0; j < d2; j++) {
				for (int k = 0; k < d3; k++) {
					for (int l = 0; l < d4; l++) {
						for (int m = 0; m < d5; m++) {
							d[cont++] = arr[i][j][k][l][m];
						}
					}
				}
			}
		}

		return this;
    }

	/**
	 * Copia todo o conteúdo do array na instância local.
	 * @param arr {@code array} base.
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor copy(float[][][][] arr) {
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
		float[] d = array();
		for (int i = 0; i < d1; i++) {
			for (int j = 0; j < d2; j++) {
				for (int k = 0; k < d3; k++) {
					for (int l = 0; l < d4; l++) {
						d[cont++] = arr[i][j][k][l];
					}
				}
			}
		}

		return this;
    }

	/**
	 * Copia todo o conteúdo do array na instância local.
	 * @param arr {@code array} base.
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor copy(float[][][] arr) {
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
		float[] d = array();
		for (int i = 0; i < d1; i++) {
			for (int j = 0; j < d2; j++) {
				for (int k = 0; k < d3; k++) {
					d[cont++] = arr[i][j][k];
				}
			}
		}

		return this;
    }

	/**
	 * Copia todo o conteúdo do array na instância local.
	 * @param arr {@code array} base.
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor copy(float[][] arr) {
        if (numDim() != 2) {
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
		float[] d = array();
		for (int i = 0; i < lin; i++) {
			for (int j = 0; j < col; j++) {
				d[id++] = arr[i][j];
			}
		}

		return this;
    }

	/**
	 * Copia todo o conteúdo do array na instância local.
	 * @param arr {@code array} base.
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor copy(float[] arr) {
        if (numDim() != 1) {
            throw new IllegalArgumentException(
                "\nTensor tem " + numDim() + " dimensões, mas deve" +
                " ter 1."
            );
        }

		if ((len() != arr.length)) {
			throw new IllegalArgumentException(
				"\nDimensões do tensor " + shapeStr() +
				" incompatíveis com as do array recebido (" + arr.length + ")."
			);
		}

		data.copy(arr);

		return this;
    }

	/**
	 * Copia apenas os dados contidos no tensor, sem levar em consideração 
	 * suas dimensões e contiguidade.
	 * <p>
	 *		Ainda é necessário que a quantidade de elementos de ambos os 
	 *		tensores sejam iguais.
	 * </p>
	 * @param t {@code Tensor} base.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor copyElements(Tensor t) {
		if (len() != t.len()) {
			throw new IllegalArgumentException(
				"\nOs tensores devem conter o mesmo número de elementos. Local = " + len() + 
				"e recebido = " + t.len()
			);
		}

		data.copy(t.data);

		return this;
	}

	/**
	 * Copia apenas os dados contidos no array, sem levar em consideração
	 * as dimensões do tensor.
	 * <p>
	 * Ainda é necessário que a quantidade de elementos do array seja igual
	 * a quantidade de elementos do tensor.
	 * </p>
	 * @param arr {@code array} base.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor copyElements(float[] arr) {
		if (arr.length != len()) {
			throw new IllegalArgumentException(
				"\nTamanho do array fornecido (" + arr.length + ") inconpatível" +
				"com os elementos do tensor (" + len() + ")."
			);
		}

		data.copy(arr);
		
		return this;
	}

	/**
	 * Adiciona o valor informado em todos os elementos do tensor.
	 * @param x valor desejado.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor add(Number x) {
		final float val = x.floatValue();
		data.add(val);
		return this;
	}

	/**
	 * Adiciona todo o conteúdo {@code elemento a elemento} usando o tensor recebido,
	 * seguindo a expressão:
	 * <pre>
	 *  this += tensor
	 * </pre>
	 * @param t {@code Tensor} base.
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor add(Tensor t) {
		if (t.len() == 1) return add(t.item());

        if (!compareShape(t)) {
            throw new IllegalArgumentException(
                "\nTensor fornecido possui shape " + t.shapeStr() +
				", shape esperado " + shapeStr()
            );
        }

		data.add(t.data);

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
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor addadd(Tensor a, Tensor b) {
		return apply(
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
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor addsub(Tensor a, Tensor b) {
		return apply(
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
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor addmul(Tensor a, Tensor b) {
		return apply(
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
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor adddiv(Tensor a, Tensor b) {
		return apply(
			this, a, b, 
			(t, t1, t2) -> t += (t1 / t2)
		);
    }

	/**
	 * Adiciona o valor informado ao conteúdo do tensor.
	 * @param x valor desejado.
	 * @param i índices desejados para adição.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor add(Number x, int... i) {
		final float val = x.floatValue();
		data.add(val, index(i));
		return this;
	}

	/**
	 * Subtrai o valor informado em todos os elementos do tensor.
	 * @param x valor desejado.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor sub(Number x) {
		final float val = x.floatValue();
		data.add(-val);
		return this;
	}

	/**
	 * Subtrai todo o conteúdo {@code elemento a elemento} usando o tensor recebido,
	 * seguindo a expressão:
	 * <pre>
	 *  this -= tensor
	 * </pre>
	 * @param t {@code Tensor} base.
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor sub(Tensor t) {
		if (t.len() == 1) return sub(t.item());

        if (!compareShape(t)) {
            throw new IllegalArgumentException(
                "\nTensor fornecido possui shape " + t.shapeStr() +
				", shape esperado " + shapeStr()
            );
        }

        data.sub(t.data);

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
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor subadd(Tensor a, Tensor b) {
		return apply(
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
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor subsub(Tensor a, Tensor b) {
		return apply(
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
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor submul(Tensor a, Tensor b) {
		return apply(
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
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor subdiv(Tensor a, Tensor b) {
		return apply(
			this, a, b, 
			(t, t1, t2) -> t -= (t1 / t2)
		);
    }

	/**
	 * Subtrai o valor informado ao conteúdo do tensor.
	 * @param x valor desejado.
	 * @param i índices desejados para adição.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor sub(Number x, int... i) {
		final float val = x.floatValue();
		data.sub(val, index(i));
		return this;
	}

	/**
	 * Multiplica o valor informado em todos os elementos do tensor.
	 * @param x valor desejado.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor mul(Number x) {
		final float val = x.floatValue();
		data.mul(val);
		return this;
	}
	
	/**
	 * Multiplica todo o conteúdo {@code elemento a elemento} usando o tensor recebido,
	 * seguindo a expressão:
	 * <pre>
	 *  this *= tensor
	 * </pre>
	 * @param t {@code Tensor} com conteúdo.
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor mul(Tensor t) {
		if (t.len() == 1) return mul(t.item());

        if (!compareShape(t)) {
            throw new IllegalArgumentException(
                "\nTensor fornecido possui shape " + t.shapeStr() +
				", shape esperado " + shapeStr()
            );
        }

		data.mul(t.data);

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
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor muladd(Tensor a, Tensor b) {
		return apply(
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
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor mulsub(Tensor a, Tensor b) {
		return apply(
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
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor mulmul(Tensor a, Tensor b) {
		return apply(
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
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor muldiv(Tensor a, Tensor b) {
		return apply(
			this, a, b, 
			(t, t1, t2) -> t *= (t1 / t2)
		);
    }

	/**
	 * Multiplica o valor informado ao conteúdo do tensor.
	 * @param x valor desejado.
	 * @param i índices desejados para adição.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor mul(Number x, int... i) {
		final float val = x.floatValue();
		data.mul(val, index(i));
		return this;
	}

	/**
	 * Divide o valor informado em todos os elementos do tensor.
	 * @param x valor desejado.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor div(Number x) {
		final float val = x.floatValue();
		data.div(val);
		return this;
	}

	/**
	 * Divide todo o conteúdo {@code elemento a elemento} usando o tensor recebido,
	 * seguindo a expressão:
	 * <pre>
	 *  this /= tensor
	 * </pre>
	 * @param t {@code Tensor} com conteúdo.
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor div(Tensor t) {
		if (t.len() == 1) return div(t.item());

        if (!compareShape(t)) {
            throw new IllegalArgumentException(
                "\nTensor fornecido possui shape " + t.shapeStr() +
				", shape esperado " + shapeStr()
            );
        }

		data.div(t.data);

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
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor divadd(Tensor a, Tensor b) {
		return apply(
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
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor divsub(Tensor a, Tensor b) {
		return apply(
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
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor divmul(Tensor a, Tensor b) {
		return apply(
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
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor divdiv(Tensor a, Tensor b) {
		return apply(this, a, b, 
			(t, t1, t2) -> t /= (t1 / t2)
		);
    }

	/**
	 * Divide o valor contido no tensor pelo valor informado.
	 * @param x valor desejado.
	 * @param i índices desejados para adição.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor div(Number x, int... i) {
		final float val = x.floatValue();
		data.div(val, index(i));
		return this;
	}

	/**
	 * Remove a dimensão desejada caso possua tamanho = 1.
	 * @param dim índice da dimensão desejada.
	 * @return {@code view} do {@code Tensor}.
	 */
	public Tensor squeeze(int dim) {
		if (dim < 0 || dim >= shape.length) {
			throw new IllegalArgumentException("\nDimensão " + dim + " inválida");
		}

		if (numDim() == 1) {
			throw new UnsupportedOperationException(
				"\nNão é possível remover a única dimensão do tensor."
			);
		}

		if (shape[dim] != 1) {
			throw new IllegalArgumentException(
				"\nDimensão dada (" + dim + ") deve ter tamanho = 1," +
				"mas tem tamanho = " + shape[dim] 
			);
		}

		int[] novoShape = new int[shape.length - 1];
		int[] novoStrides = new int[stride.length - 1];

		int id = 0;
		for (int i = 0; i < shape.length; i++) {
			if (i != dim) {
				novoShape[id] = shape[i];
				novoStrides[id] = stride[i];
				id++;
			}
		}

		return new Tensor(data, novoShape, novoStrides);// view
	}

	/**
	 * Adiciona uma nova dimensão com tamanho = 1.
	 * @param dim índice da dimensão que será adicionada.
	 * @return {@code view} do {@code Tensor}.
	 */
    public Tensor unsqueeze(int dim) {
		if (dim < 0 || dim > shape.length) {
			throw new IllegalArgumentException("\nDimensão " + dim + " inválida");
		}

		int n = numDim();
		int[] novoShape = new int[n + 1];
		int[] novoStrides = new int[stride.length + 1];

		for (int i = 0; i < dim; i++) {
			novoShape[i] = shape[i];
			novoStrides[i] = stride[i];
		}

		novoShape[dim] = 1;
		novoStrides[dim] = (dim < stride.length) ? stride[dim] : 1;

		for (int i = dim; i < n; i++) {
			novoShape[i + 1] = shape[i];
			novoStrides[i + 1] = stride[i];
		}


		return new Tensor(data, novoShape, novoStrides);// view
    }

	/**
	 * Achata os dados do tensor.
	 * @return se o {@code Tensor} for {@code contíguo}, retorna uma {@code view},
	 * caso contrário retorna uma {@code cópia} achatada do tensor.
	 */
	public Tensor flatten() {
		if (isContiguous()) {
			return new Tensor(
				data,
				new int[]{ len() },
				new int[]{ 1 }
			);
		}

		return new Tensor(len()).copy(this);
	}

    /**
     * Realiza uma multiplicação elemento a elemento entre {@code A} e 
     * {@code B}, junto da multiplicação de um escalar {@code Alfa} e acumula
     * o resultado na instância local.
     * <p>
     *      Equivalente a:
     * </p>
     * <pre>
     * this += alfa * (A * B)
     * </pre>
     * Essa função foi inspirada no {@code PyTorch}:
     * @see <a href="https://docs.pytorch.org/docs/stable/generated/torch.addcmul.html">
     *      torch.addcdiv (PyTorch) </a>
     * @param a {@code Tensor} A.
     * @param b {@code Tensor} B.
     * @param alpha {@code valor} escalar multiplicativo.
	 * @return {@code Tensor} local alterado.
     */
	public Tensor addcmul(Tensor a, Tensor b, float alpha) {
		data.addcmul(a.data, b.data, alpha);
		return this;
	}

    /**
     * Realiza uma divisão elemento a elemento entre {@code A} e 
     * {@code B}, junto da multiplicação de um escalar {@code Alfa} e acumula
     * o resultado na instância local.
     * <p>
     *      Equivalente a:
     * </p>
     * <pre>
     * this += alfa * (A / B)
     * </pre>
     * Essa função foi inspirada no {@code PyTorch}:
     * @see <a href="https://docs.pytorch.org/docs/stable/generated/torch.addcdiv.html">
     *      torch.addcdiv (PyTorch)
     *      </a>
     * @param a {@code Tensor} numerador.
     * @param b {@code Tensor} denominador.
     * @param alpha {@code valor} escalar multiplicativo.
	 * @return {@code Tensor} local alterado.
     */
	public Tensor addcdiv(Tensor a, Tensor b, float alpha) {
		data.addcdiv(a.data, b.data, alpha);
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
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor apply(FloatUnaryOperator fun) {
		data.apply(fun);
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
	 * @param t {@code Tensor} base.
	 * @param fun função para aplicar no tensor base.
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor apply(Tensor t, FloatUnaryOperator fun) {
		if (!compareShape(t)) {
			throw new IllegalArgumentException(
				"\nAs dimensões do tensor fornecido " + t.shapeStr() +
				" e as da instância local " + shapeStr() + " devem ser iguais."
			);
		}

		data.apply(t.data, fun);
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
	 * @return {@code Tensor} local alterado.
	 */
    public Tensor apply(Tensor a, Tensor b, FloatBinaryOperator fun) {
		if (!compareShape(a) || !compareShape(b)) {
			throw new IllegalArgumentException(
				"\nAs dimensões dos tensores A " + a.shapeStr() +
				", B " + b.shapeStr() +
				" e as da instância local " + shapeStr() + " devem ser iguais."
			);
		}
		
		final int n = len();
		float[] td = array();
		float[] da = a.array();
		float[] db = b.array();

		for (int i = 0; i < n; i++) {
			td[i] = fun.apply(da[i], db[i]);
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
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor apply(Tensor a, Tensor b, Tensor c, FloatTernaryOperator fun) {
		if (!compareShape(a) || !compareShape(b) || !compareShape(c)) {
			throw new IllegalArgumentException(
				"\nAs dimensões dos tensores A " + a.shapeStr() +
				", B " + b.shapeStr() +
				" e C " + c.shapeStr() +
				" e as da instância local " + shapeStr() + " devem ser iguais."
			);
		}

		final int n = len();
		float[] td = array();
		float[] da = a.array();
		float[] db = b.array();
		float[] dc = c.array();

		for (int i = 0; i < n; i++) {
			td[i] = fun.apply(da[i], db[i], dc[i]);
		}
		
		return this;
	}

	/**
	 * Retorna o valor contido no tensor, caso ele possua apenas um elemento.
	 * @return valor contido no tensor.
	 */
	public float item() {
		if (len() != 1) {
			throw new IllegalArgumentException(
				"\nO tensor deve conter apenas um elemento."
			);
		}

		return data.get(0);
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
	 * @return novo {@code Tensor} contendo o resultado.
	 */
	public Tensor map(FloatUnaryOperator fun) {
		Tensor t = clone();		
		return t.apply(this, fun);
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
	 * @param t segundo {@code Tensor} para aplicar a função.
	 * @param fun função desejada.
	 * @return novo {@code Tensor} contendo o resultado.
	 */
	public Tensor map(Tensor t, FloatBinaryOperator fun) {
		if (!compareShape(t)) {
			throw new IllegalArgumentException(
				"\nTensor " + t.shapeStr() + " deve conter mesmo formato do " +
				"tensor local " + shapeStr()
			);
		}

		Tensor map = new Tensor(shape());
		map.apply(this, t, fun);

		return map;
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
	 * @param start valor inicial.
	 * @param fun função desejada.
	 * @return novo {@code Tensor} contendo o resultado.
	 */
	public Tensor reduce(Number start, FloatBinaryOperator fun) {
		float res = start.floatValue();

		float[] d = array();
		for (float val : d) {
			res = fun.apply(res, val);
		}

		return new Tensor(1).set(res, 0);
	}

	/**
	 * Retorna um {@code Tensor} contendo a soma dos elementos da 
     * instância local.
	 * @return novo {@code Tensor} contendo o resultado.
	 */
    public Tensor sum() {
		float soma = data.sum();
		return new Tensor(1).set(soma, 0);
    }

	/**
	 * Retorna um {@code Tensor} contendo a média aritmética dos 
     * elementos da instância local.
	 * @return novo {@code Tensor} contendo o resultado.
	 */
	public Tensor mean() {
		float med = data.mean();
		return new Tensor(1).set(med, 0);
    }

	/**
	 * Retorna um {@code Tensor} contendo o valor máximo dentro dos 
     * elementos da instância local.
	 * @return novo {@code Tensor} contendo o resultado.
	 */
	public Tensor max() {
		float max = data.max();
		return new Tensor(1).set(max, 0);
	}

	/**
	 * Retorna um {@code Tensor} contendo o valor mínimo dentro dos 
     * elementos da instância local.
	 * @return novo {@code Tensor} contendo o resultado.
	 */
	public Tensor min() {
		float min = data.min();
		return new Tensor(1).set(min, 0);
	}

	/**
	 * Retorna um {@code Tensor} contendo o desvio padrão de acordo com os
     * elementos da instância local.
	 * @return novo {@code Tensor} contendo o resultado.
     */
	public Tensor std() {
		float desvp = data.std();
		return new Tensor(1).set(desvp, 0);
	}

	/**
	 * Restringe o conteúdo de dados do tensor entre um valor 
	 * mínimo e máximo.
	 * @param min valor mínimo.
	 * @param max valor máximo.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor clamp(float min, float max) {
		data.clamp(min, max);
		return this;
	}

	/**
	 * Normaliza os valores do tensor dentro do intervalo especificado.
	 * @param min valor mínimo do intervalo.
	 * @param max valor máximo do intervalo.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor normMinMax(Number min, Number max) {
		float valMin = min().item();
		float valMax = max().item();

		float intOrig = valMax - valMin;
		float intNovo = max.floatValue() - min.floatValue();

		return apply(
			x -> ((x - valMin) / intOrig) * intNovo + min.floatValue()
		);
	}

	/**
	 * Aplica a função de ativação {@code ReLU} em todos os
	 * elementos do tensor.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor relu() {
		return apply(x -> x > 0.0f ? x : 0.0f);
	}

	/**
	 * Aplica a função de ativação {@code Sigmoid} em todos os
	 * elementos do tensor.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor sigmoid() {
		return apply(x -> 1.0f / (float) (1.0 + Math.exp(-x)));
	}

	/**
	 * Aplica a função de ativação {@code TanH} (Tangente Hiperbólica)
	 * em todos os elementos do tensor.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor tanh() {
		return apply(x -> 2.0f / (float) (1.0 + Math.exp(-2 * x)) - 1.0f);
	}

	/**
	 * Aplica a função de ativação {@code Atan} (Arco Tangente)
	 * em todos os elementos do tensor.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor atan() {
		return apply(x -> (float) Math.atan(x));
	}

	/**
	 * Calcula o valor {@code seno} de todos os elementos do tensor.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor sin() {
		return apply(x -> (float) Math.sin(x));
	}

	/**
	 * Calcula o valor {@code cosseno} de todos os elementos do tensor.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor cos() {
		return apply(x -> (float) Math.cos(x));
	}

	/**
	 * Calcula o valor {@code tangente} de todos os elementos do tensor.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor tan() {
		return apply(x -> (float) Math.tan(x));
	}

	/**
	 * Calcula o valor {@code absoluto} de cada elemento do do tensor.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor abs() {
		return apply(x -> Math.abs(x));
	}

	/**
	 * Calcula o valor {@code exponencial} de cada elemento do do tensor.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor exp() {
		return apply(x -> (float) Math.exp(x));
	}

	/**
	 * Calcula o valor {@code logaritmo natural} de cada elemento do do tensor.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor log() {
		return apply(x -> (float) Math.log(x));
	}

	/**
	 * Retorna a quantidade total de elementos no tensor.
	 * @return número elementos do tensor.
	 */
	public int len() {
		return data.len();
	}

	/**
	 * Retorna o tamanho da dimensão especifidada.
	 * @param dim dimensão desejada.
	 * @return tamanho da dimensão.
	 */
	public int lenDim(int dim) {
		if (dim < 0 || dim > shape.length) {
			throw new IllegalArgumentException(
				"\nDimensão " + dim + " inválida para tensor " + numDim() + "D."
			);
		}

		return shape[dim];
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
	 * Retorna um array contendo os strides do tensor.
	 * @return strides do tensor.
	 */
	public int[] strides() {
		return stride.clone();
	}

	/**
	 * Retorna o offset dos dados do tensor.
	 * <p>
	 *		Essa informação se torna relevante a medida
	 *		que o tensor é uma view de outro tensor.
	 * </p>
	 * @return offset de dados do tensor.
	 */
	public int offset() {
		return data.offset();
	}

	/**
	 * Função interna para transformar arrays em strings (shape e stride).
	 * @param arr {@code array} base.
	 * @return
	 */
	private String arrayStr(int[] arr) {
        StringBuilder sb = new StringBuilder();

        sb.append("(");
		sb.append(arr[0]);
        for (int i = 1; i < arr.length; i++) {
            sb.append(", ").append(arr[i]);
        }
        sb.append(")");

        return sb.toString();
	}

	/**
	 * Retorna uma String contendo as dimensões do tensor.
	 * @return dimensões do tensor em formato de texto.
	 */
    public String shapeStr() {
		return arrayStr(shape);
    }

	/**
	 * Retorna uma String contendo os strides do tensor.
	 * @return strides do tensor em formato de texto.
	 */
    public String strideStr() {
		return arrayStr(stride);
    }

	/**
	 * Compara todo o conteúdo da instância local, isso inclui as {@code dimensões}
	 * de cada tensor e seus {@code elementos individuais}.
	 * <p>
	 *		Este método de comparação não é recomendado se os valores dos tensores
	 *		forem muito sensíveis (com muitas cadas decimais), pois compara diretamente
	 *		um valor com outro, o que pode não ser útil em aplicações específicas.
	 * </p>
	 * <p>
	 * 		Para comparações com tolerância, use:
	 * </p>
	 * <pre>
	 *float eps = ... //tolerância
	 *t.comp(t2, eps)
	 * </pre>
	 * @param t {@code Tensor} base.
	 * @return {@code true} caso sejam iguais, {@code false} caso contrário.
	 */
	public boolean compare(Tensor t) {
		return compare(t, 0f);
	}

	/**
	 * Compara todo o conteúdo da instância local, isso inclui as {@code dimensões}
	 * de cada tensor e seus {@code elementos individuais}.
	 * <p>
	 * 		A comparação entre os valores se utiliza de um valor de tolerância, para
	 * 		situações mais sensíveis.
	 * </p>
	 * @param t {@code Tensor} base.
	 * @param eps valor de tolerância.
	 * @return {@code true} caso sejam iguais, {@code false} caso contrário.
	 */
	public boolean compare(Tensor t, float eps) {
		if (!compareShape(t)) return false;

		final int n = len();
		float[] d = data.data();
		float[] td = t.data.data();

		final int off1 = offset();
		final int off2 = t.offset(); 
		
		for (int i = 0; i < n; i++) {
			float a =  d[off1 + i];
			float b = td[off2 + i];
			if (Math.abs(a - b) > eps) return false;
		}

		return true;
	}

    /**
     * Verifica se o shape do tensor fornecido é igual ao shape
     * da instância local.
     * @param t {@code Tensor} desejado.
     * @return {@code true} caso as dimensões de ambos os tensores sejam
     * iguais, {@code false} caso contrário.
     */
    public boolean compareShape(Tensor t) {
        int n = shape.length;
        if (n != t.shape.length) return false;

        for (int i = 0; i < n; i++) {
            if (shape[i] != t.shape[i]) return false;
        }

        return true;
    }

    /**
     * Calcula o tamanho em {@code bytes} do tensor, 
     * levando em consideração a arquitetura da JVM (32 ou 64 bits).
     * @return tamanho em bytes.
     */
	public long sizeBytes() {
		String jvmBits = System.getProperty("sun.arch.data.model");
        long bits = Long.valueOf(jvmBits);

        long tamObj;
		// overhead da jvm
        if (bits == 32) tamObj = 8;
        else if (bits == 64) tamObj = 16;
        else throw new IllegalStateException(
            "\nSem suporte para plataforma de " + bits + " bits."
        );

		long tamVars = data.sizeBytes();
		long tamShape = shape.length * 4; // int = 4 bytes
		long tamStride = stride.length * 4; // int = 4 bytes
		return tamObj + tamVars + tamShape + tamStride;
	}

	/**
	 * Retorna o conjunto de dados do tensor.
	 * <p>
	 *		Essa função expõe os dados internos do tensor, é
	 *		recomendável não fazer alterações neste conjunto de 
	 *		dados.
	 * </p>
	 * @return Dados do tensor.
	 */
	public TensorData data() {
		return data;
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
	public float[] array() {
		return data.data();
	}

	/**
	 * Configura o nome do tensor.
	 * @param name novo nome.
	 * @return {@code Tensor} local alterado.
	 */
	public Tensor name(String name) {
		if (name != null) {
			name = name.trim();
			if (!name.isEmpty()) this.name = name;
		}

		return this;
	}

	/**
	 * Retorna o nome do tensor.
	 * @return nome do tensor.
	 */
	public String name() {
		return this.name;
	}

    @Override
    public String toString() {
        return buildPrint();
    }

	/**
	 * Exibe, {@code via terminal}, todo o conteúdo do tensor.
	 */
	public void print() {
		System.out.println(buildPrint());
	}

	/**
	 * Monta as informações de exibição do tensor.
	 * @return string formatada.
	 */
    private String buildPrint() {
		// desse jeito aqui o array usado vai conter
		// só os dados que o tensor aponta, se for uma view
		// vai usar apenas o subconjunto dessa view
		float arr[] = data.toArray();
		
		int maxCasasDecimais = 0;
		for (float valor : arr) {
			String valorStr = Float.toString(valor);
			int pontoIndex = valorStr.indexOf('.');
			int decimais = (pontoIndex >= 0) ? valorStr.length() - pontoIndex - 1 : 0;
			if (decimais > maxCasasDecimais) maxCasasDecimais = decimais;
		}

		int tamMaximo = -1;
		for (float valor : arr) {
			String valorStr = strVal(valor, maxCasasDecimais);
			int tamValor = valorStr.length();
			if (tamValor > tamMaximo) tamMaximo = tamValor;
		}

		StringBuilder sb = new StringBuilder();
		
		String cabecalho = name() + " (";
		sb.append(cabecalho);

		int[] indices = new int[shape.length];
		
		buildPrintRecursive(sb, indices, 0, tamMaximo, maxCasasDecimais, cabecalho.length());

		sb.append(")");
		return sb.toString();
	}

	/**
	 * Monta as informações do tensor.
	 * @param sb buffer.
	 * @param indices índices dos dados.
	 * @param dim dimensão atual.
	 * @param maxTam tamanho máximo.
	 * @param maxDecimal tamanho máximo de casas decimais.
	 * @param padBase espaçamento base (padding).
	 */
	private void buildPrintRecursive(StringBuilder sb, int[] indices, int dim, int maxTam, int maxDecimal, int padBase) {
		sb.append("[");
		
		int tamDim = shape[dim];
		boolean ultimaDim = (dim == shape.length - 1);

		for (int i = 0; i < tamDim; i++) {
			indices[dim] = i;

			if (ultimaDim) {
				if (i > 0) sb.append(", ");
				String valorStr = strVal(get(indices), maxDecimal);
				sb.append(" ".repeat(maxTam - valorStr.length())).append(valorStr);
				
			} else {
				
				if (i > 0) {
					sb.append(",");
					int quebraLinha = shape.length - dim - 1;
					sb.append("\n".repeat(quebraLinha));
					sb.append(" ".repeat(padBase + dim + 1));
				}
				
				buildPrintRecursive(sb, indices, dim + 1, maxTam, maxDecimal, padBase);
			}
		}

		sb.append("]");
	}

	/**
	 * Retorna o valor numérico formatado em string.
	 * @param x valor numérico. 
	 * @param precision quantidade de casas decimais desejada.
	 * @return valor formatado.
	 */
	private String strVal(float x, int precision) {
		return String.format("%.0" + precision + "f", x).replace(',', '.');
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
			clone.stride = stride.clone();
			
			clone.data = new TensorData(this.data.toArray());// lidar com views
			
			return clone;
		} catch (CloneNotSupportedException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public boolean equals(Object obj) {
		return (obj instanceof Tensor) && compare((Tensor) obj);
	}

	@Override
	public Iterator<Float> iterator() {
		return new TensorIterator();
	}

	/**
	 * Iterador para usar com o tensor, usando para percorrer
	 * seuss elementos sequencialmente.
	 */
	class TensorIterator implements Iterator<Float> {
		private int linear = 0;
		private int total = len();
		private int numDim = numDim();
		private int[] coords = new int[shape.length];

		@Override
		public boolean hasNext() {
			return linear < total;
		}

		@Override
		public Float next() {
			if (!hasNext()) throw new NoSuchElementException(
				"\nSem elementos para iterar."
			);

			int ajuste = 0;
			for (int i = 0; i < numDim; i++) {
				ajuste += coords[i] * stride[i];
			}

			for (int i = numDim - 1; i >= 0; i--) {
				coords[i]++;
				if (coords[i] < shape[i]) break;
				coords[i] = 0;
			}

			linear++;
			return data.get(ajuste);
		}
	}

	/**
	 * Exporta os dados do tensor em um arquivo externo.
	 * <h4>
	 *		Observação
	 * </h4>
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
	 * @param path caminho de destino.
	 */
	public void save(String path) {
		SerialTensor.salvar(this, path);
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
		if (len() == 1) {
			throw new UnsupportedOperationException(
				"\nNão é possível obter um subtensor a partir de um tensor escalar."
			);
		}

		if (dim < 0 || dim >= shape[0]) {
			throw new IllegalArgumentException(
				"\nÍndice " + dim + " inválido."
			);
		}

		// escalar
		if (numDim() == 1) {
			return new Tensor(1).set(data.get(dim), 0);
		}

		int[] novoShape = Arrays.copyOfRange(shape, 1, shape.length);
		int stride = calcLen(novoShape);

		TensorData subdados = data.view(dim * stride, stride);
		return new Tensor(subdados, novoShape, initStrides(novoShape));
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
	public Tensor broadcast(Tensor t, FloatBinaryOperator op) {
		int[] outShape = broadcastShape(this.shape, t.shape);
		Tensor out = new Tensor(outShape);

		int n = out.len();

		int[] stridesA = initStrides(this.shape);
		int[] stridesB = initStrides(t.shape);

		int desvioA = this.shape.length < outShape.length ? outShape.length - this.shape.length : 0;
		int desvioB = t.shape.length < outShape.length ? outShape.length - t.shape.length : 0;

		float[] data = array();
		float[] td = t.array();
		float[] bd = out.array();

		for (int ids = 0; ids < n; ids++) {
			int temp = ids;

			int idA = 0, idB = 0;
			for (int d = outShape.length - 1; d >= 0; d--) {
				int cord = temp % outShape[d];
				temp /= outShape[d];

				if (d - desvioA >= 0) {// ajuste id de A
					int dimA = this.shape[d - desvioA];
					if (dimA != 1) {
						idA += cord * stridesA[d - desvioA];
					}
				}

				if (d - desvioB >= 0) {// ajuste id de B
					int dimB = t.shape[d - desvioB];
					if (dimB != 1) {
						idB += cord * stridesB[d - desvioB];
					}
				}
			}

			bd[ids] = op.apply(data[idA], td[idB]);
		}

		return out;
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
					+ "A = " +arrayStr(shapeA) + " e B = " + arrayStr(shapeB)
				);
			}
		}

		return broadShape;
	}

	/**
	 * Verifica se o tensor é contíguo na memória.
	 * <p>
	 *		Um tensor é considerado contíguo se os strides correspondem ao layout 
	 *		row-major por padrão, em outras palavras, se os elementos estão armazenados 
	 *		de forma sequencial sem gaps entre dimensões.
	 * </p>
	 * @return resultado da verificação.
	 */
	public boolean isContiguous() {
		int strideEsperado = 1;

		for (int i = shape.length - 1; i >= 0; i--) {
			if (stride[i] != strideEsperado) return false;
			strideEsperado *= shape[i];
		}

		return true;
	}

	/**
	 * Retorna uma versão contígua do tensor, baseada na estrutura row-major.
	 * <p>
	 *		Caso o tensor já seja contíguo, retorna ele mesmo.
	 * </p>
	 * @return tensor contíguo.
	 */
	public Tensor contiguous() {
		if (isContiguous()) return this;
		
		final float[] dados = array();
		final int offset = offset();

		Tensor contg = new Tensor(shape);
		float[] dst = contg.array();

		int ndim = shape.length;
		int tam = contg.len();
		int[] ids = new int[ndim];

		for (int linear = 0; linear < tam; linear++) {
			int idBase = offset;
			for (int d = 0; d < ndim; d++) {
				idBase += ids[d] * stride[d];
			}

			dst[linear] = dados[idBase];

			for (int d = ndim - 1; d >= 0; d--) {
				ids[d]++;
				if (ids[d] < shape[d]) break;
				ids[d] = 0;
			}
		}

		return contg;
	}

	/**
	 * Retorna a soma dos elementos ao longo de um eixo específico.
	 * @param axis eixo ao longo do qual será feita a soma.
	 * @return {@code Tensor} resultante com o eixo reduzido.
	 */
	public Tensor sum(int axis) {
		if (axis < 0 || axis >= shape.length) {
			throw new IllegalArgumentException(
				"\nEixo (" + axis + ") inválido."
			);
		}

		int[] resShape = new int[shape.length - 1];
		for (int i = 0, j = 0; i < shape.length; i++) {
			if (i != axis) resShape[j++] = shape[i];
		}

		Tensor res = new Tensor(resShape);

		int[] idIn  = new int[shape.length];
		int[] idRes = new int[resShape.length];
		float[] rd = res.array();

		for (int i = 0; i < res.len(); i++) {
			linearIndex(i, resShape, idRes);

			for (int j = 0, k = 0; j < shape.length; j++) {
				if (j == axis) continue;
				idIn[j] = idRes[k++];
			}

			float soma = 0.0f;
			for (int a = 0; a < shape[axis]; a++) {
				idIn[axis] = a;
				soma += get(idIn);
			}

			rd[i] = soma;
		}

		return res;
	}

	/**
	 * Converte um índice linear em um array de coordenadas multidimensionais.
	 * <p>
	 *		Essencialmente essa função atua como {@code unravelIndex}, em implementações
	 *		de bibliotecas como NumPy e PyTorch.
	 * </p>
	 * @param id índice base.
	 * @param shape {@code array} com formato do tensor.
	 * @param coords {@code array} de destino.
	 * @see {@link {@code NumPy} https://numpy.org/devdocs/reference/generated/numpy.unravel_index.html}
	 * @see {@link {@code PyTorch} https://docs.pytorch.org/docs/stable/generated/torch.unravel_index.html}
	 */
	private void linearIndex(int id, int[] shape, int[] coords) {
		for (int i = shape.length - 1; i >= 0; i--) {
			coords[i] = id % shape[i];
			id /= shape[i];
		}
	}

	/**
	 * Retorna o índice onde existe o maior valor em uma dimensão especificada.
	 * @param axis dimensão desejada.
	 * @return novo {@code Tensor}.
	 */
	public Tensor argmax(int axis) {
		if (axis < 0 || axis >= shape.length) {
			throw new IllegalArgumentException(
				"\nEixo (" + axis + ") inválido."
			);
		}

		if (shape.length == 1) {
			int arg = 0;
			float maxVal = Float.NEGATIVE_INFINITY;
			for (int i = 0; i < shape[0]; i++) {
				float v = data.get(i);
				if (v > maxVal) {
					maxVal = v;
					arg = i;
				}
			}

			Tensor res = new Tensor(1);
			res.set(arg, 0);
			return res;
		}

		int[] shapeRes = new int[shape.length - 1];
		for (int i = 0, j = 0; i < shape.length; i++) {
			if (i != axis) shapeRes[j++] = shape[i];
		}

		Tensor res = new Tensor(shapeRes);

		int[] idIn  = new int[shape.length];
		int[] idRes = new int[shapeRes.length];
		float[] rd = res.array();

		for (int i = 0; i < res.len(); i++) {
			linearIndex(i, shapeRes, idRes);

			for (int j = 0, k = 0; j < shape.length; j++) {
				if (j == axis) continue;
				idIn[j] = idRes[k++];
			}

			float maxVal = Float.NEGATIVE_INFINITY;
			int maxIdx = 0;

			for (int a = 0; a < shape[axis]; a++) {
				idIn[axis] = a;
				float v = get(idIn);

				if (v > maxVal) {
					maxVal = v;
					maxIdx = a;
				}
			}

			rd[i] = maxIdx;
		}

		return res;
	}

}