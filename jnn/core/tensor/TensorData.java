package jnn.core.tensor;

import java.util.Arrays;

/**
 * <h2>
 *      Conjunto de Dados de um Tensor
 * </h2>
 * O TensorData é um conteiner para um array de elementos contíguo em 
 * memória com a promessa de otimizar o desempenho de operações com
 * Tensores.
 * @see {@link jnn.core.tensor.Tensor}
 */
public class TensorData {

    /**
     * Array de elementos.
     */
    private double[] dados;

    /**
     * Índice inicial. (para views)
     */
    private int offset;

    /**
     * Tamanho do conjunto de dados.
     * <p>
     *      Pode conter tamanhos diferentes se
     *      o conjunto for uma view de outro TensorData.
     * </p>
     */
    private int tam; 

    /**
     * Inicializa um {@code TensorData} a partir de dados bem definidos.
     * <p>
     *      O conteúdo referenciado como o mesmo do array.
     * </p>
     * @param arr {@code array} base.
     * @param offset indice inicial a partir do array base.
     * @param tam tamanho final do conjunto de elementos.
     */
    public TensorData(double[] arr, int offset, int tam) {
        if (arr == null) {
            throw new IllegalArgumentException(
                "\nArray nulo."
            );
        }
        if (offset < 0 || tam < 0) {
            throw new IllegalArgumentException(
                "\nOffset e tamanho devem ser maiores que zero."
            );
        }
        if (offset + tam > arr.length) {
            throw new IllegalArgumentException(
                "\nOffset + tamanho excede o tamanho do array. (" +
                (offset + tam) + " > " + arr.length + ")"
            );
        }

        this.offset = offset;
        this.tam = tam;
        this.dados = arr;
    }

    /**
     * Inicializa um {@code TensorData} a partir de um array.
     * <p>
     *      O conteúdo estará copiado.
     * </p>
     * @param arr {@code array} base.
     */
    public TensorData(double[] arr) {
        if (arr == null) {
            throw new IllegalArgumentException(
                "\nArray nulo."
            );
        }

        this.dados = Arrays.copyOf(arr, arr.length);
        this.offset = 0;
        this.tam = arr.length;
    }

    /**
     * Inicializa um {@code TensorData} a partir um tamanho especificado.
     * <p>
     *      O conteúdo estará zerado.
     * </p>
     * @param arr {@code array} base.
     */
    public TensorData(int tam) {
        if (tam < 1) {
            throw new IllegalArgumentException(
                "\nTamanho deve ser maior que zero, mas recebido " + tam + "."
            );
        }

        this.dados = new double[tam];
        this.offset = 0;
        this.tam = dados.length;
    }

    /**
     * Retorna um novo {@code TensorData} que é uma visualização a partir
     * da instância local.
     * @param inicio índicide inicial do conjunto de elementos.
     * @param tam tamamnho do conjunto de elementos.
     * @return view da instância local.
     */
    public TensorData view(int inicio, int tam) {
        if (inicio < 0 || inicio + tam > dados.length) {
            throw new IllegalArgumentException("\nView fora do limite do tensor data.");
        }
        return new TensorData(dados, offset + inicio, tam);
    }

    /**
     * Retorna o valor contido no conjunto de dados.
     * @param id índice linear baseado no array de dados.
     * @return valor obtido.
     */
    public double get(int id) {
        return dados[offset + id];
    }

    /**
     * Altera o valor contido no conjunto de dados.
     * @param x valor base.
     * @param id índice linear baseado no array de dados.
     */
    public void set(Number x, int id) {
        dados[offset + id] = x.doubleValue();
    }

    /**
     * Preenche todo o conjunto de dados.
     * @param x valor base.
     */
    public void preencher(Number x) {
        final double val = x.doubleValue();
        Arrays.fill(dados, offset, dados.length, val);
    }

    /**
     * Zera os valores do conjunto de dados.
     */
    public void zero() {
        preencher(0);
    }

    /**
     * Copia o conteúdo do array para o conjunto de dados.
     * @param arr {@code array} base.
     */
    public void copiar(double[] arr) {
        final int n = dados.length;
        System.arraycopy(arr, 0, dados, 0, n);
    }

    /**
     * Copia o conteúdo de outro conjunto de dados para a isntância local.
     * @param td {@code TensorData} base.
     */
    public void copiar(TensorData td) {
        copiar(td.dados);
    }
    
    /**
     * Adiciona um valor em todo o conjunto de dados.
     * @param x valor base.
     */
    public void add(double x) {
        final int n = tam();
        for (int i = 0; i < n; i++) {
            dados[offset + i] += x;
        }
    }
    
    /**
     * Subtrai um valor em todo o conjunto de dados.
     * @param x valor base.
     */
    public void sub(double x) {
        add(-x);
    }
 
    /**
     * Multiplica um valor em todo o conjunto de dados.
     * @param x valor base.
     */
    public void mul(double x) {
        final int n = tam();
        for (int i = 0; i < n; i++) {
            dados[offset + i] *= x;
        }
    }
    
    /**
     * Divide um valor em todo o conjunto de dados.
     * @param x valor base.
     */
    public void div(double x) {
        final int n = tam();
        for (int i = 0; i < n; i++) {
            dados[offset + i] /= x;
        }
    }

    /**
     * Adiciona um valor de acordo com um índice linear.
     * @param x valor base.
     * @param id índice baseado no array do conjunto de elementos.
     */
    public void add(double x, int id) {
        if (id < 0 || id > tam) {
            throw new IllegalArgumentException(
                "\nÍndice " + id + " inválido."
            );
        }

        dados[offset + id] += x;
    }

    /**
     * Subtrai um valor de acordo com um índice linear.
     * @param x valor base.
     * @param id índice baseado no array do conjunto de elementos.
     */
    public void sub(double x, int id) {
        add(-x, id);
    }

    /**
     * Multiplica um valor de acordo com um índice linear.
     * @param x valor base.
     * @param id índice baseado no array do conjunto de elementos.
     */
    public void mul(double x, int id) {
        if (id < 0 || id > tam) {
            throw new IllegalArgumentException(
                "\nÍndice " + id + " inválido."
            );
        }

        dados[offset + id] *= x;
    }

    /**
     * Divide um valor de acordo com um índice linear.
     * @param x valor base.
     * @param id índice baseado no array do conjunto de elementos.
     */
    public void div(double x, int id) {
        if (id < 0 || id > tam) {
            throw new IllegalArgumentException(
                "\nÍndice " + id + " inválido."
            );
        }

        dados[offset + id] /= x;
    }

    /**
     * Adiciona o conteúdo do array ao conjunto de dados.
     * @param arr {@code array} base.
     */
    public void add(double[] arr) {
        final int n = tam();
        if (arr.length != n) {
            throw new IllegalArgumentException(
                "\nTamanho do array (" + arr.length + ") deve ser igual ao tamanho de dados (" + n + ")."
            );
        }

        for (int i = 0; i < n; i++) {
            dados[offset + i] += arr[i];
        }
    }

    /**
     * Subtrai o conteúdo do array ao conjunto de dados.
     * @param arr {@code array} base.
     */
    public void sub(double[] arr) {
        final int n = tam();
        if (arr.length != n) {
            throw new IllegalArgumentException(
                "\nTamanho do array (" + arr.length + ") deve ser igual ao tamanho de dados (" + n + ")."
            );
        }

        for (int i = 0; i < n; i++) {
            dados[offset + i] -= arr[i];
        }
    }
    
    /**
     * Multiplica o conteúdo do array ao conjunto de dados.
     * @param arr {@code array} base.
     */
    public void mul(double[] arr) {
        final int n = tam();
        if (arr.length != n) {
            throw new IllegalArgumentException(
                "\nTamanho do array (" + arr.length + ") deve ser igual ao tamanho de dados (" + n + ")."
            );
        }

        for (int i = 0; i < n; i++) {
            dados[offset + i] *= arr[i];
        }
    }
    
    /**
     * Divide o conteúdo do array ao conjunto de dados.
     * @param arr {@code array} base.
     */
    public void div(double[] arr) {
        final int n = tam();
        if (arr.length != n) {
            throw new IllegalArgumentException(
                "\nTamanho do array (" + arr.length + ") deve ser igual ao tamanho de dados (" + n + ")."
            );
        }

        for (int i = 0; i < n; i++) {
            dados[offset + i] /= arr[i];
        }
    }

    /**
     * Retorna o array do conjunto de dados.
     * @return referência do conjunto de dados.
     */
    public double[] data() {
        return dados;
    }

    /**
     * Retorna o tamanho do conjunto de dados.
     * @return tamanho do conjunto de dados.
     */
    public int tam() {
        return tam;
    }

    /**
     * Retorna o índice inicial de contagem do conjunto de dados.
     * @return offset do array de elementos.
     */
    public int offset() {
        return offset;
    }

    /**
     * Retorna o tamanho em memória aproximado (em bytes) do objeto TensorData.
     * @return tamanho em bytes aproximado.
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

        long tamArr = 8 * tam();// double 8 bytes
        long tamOffset = 4;// int 4 bytes
        long tamTam = 4;// int 4 bytes
		return tamObj + tamArr + tamOffset + tamTam;
    }

    @Override
    public TensorData clone() {
        return new TensorData(dados.length);
    }
}
