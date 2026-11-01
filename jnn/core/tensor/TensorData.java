package jnn.core.tensor;

import java.util.Arrays;

import jnn.core.tensor.operadores.FloatUnaryOperator;

/**
 * <h2>
 *      Conjunto de Dados de um Tensor
 * </h2>
 * O TensorData é um conteiner para um array de elementos contíguo em 
 * memória com a promessa de otimizar o desempenho de operações com
 * Tensores.
 * @see jnn.core.tensor.Tensor Tensor
 */
public class TensorData {

    /**
     * Array de elementos.
     */
    private final float[] data;

    /**
     * Índice inicial. (para views)
     */
    private final int offset;

    /**
     * Tamanho do conjunto de dados.
     * <p>
     *      Pode conter tamanhos diferentes se
     *      o conjunto for uma view de outro TensorData.
     * </p>
     */
    private final int len; 

    /**
     * Inicializa um {@code TensorData} a partir de dados bem definidos.
     * <p>
     *      O conteúdo é referenciado como o mesmo do array.
     * </p>
     * @param arr {@code array} base.
     * @param offset indice inicial a partir do array base.
     * @param len tamanho final do conjunto de elementos.
     */
    public TensorData(float[] arr, int offset, int len) {
        if (arr == null) {
            throw new IllegalArgumentException(
                "\nArray nulo."
            );
        }
        if (offset < 0 || len < 0) {
            throw new IllegalArgumentException(
                "\nOffset e tamanho devem ser maiores que zero."
            );
        }
        if (offset + len > arr.length) {
            throw new IllegalArgumentException(
                "\nOffset + tamanho excede o tamanho do array. (" +
                (offset + len) + " > " + arr.length + ")"
            );
        }

        this.offset = offset;
        this.len = len;
        this.data = arr;
    }

    /**
     * Inicializa um {@code TensorData} a partir de um array.
     * <p>
     *      O conteúdo estará copiado.
     * </p>
     * @param arr {@code array} base.
     */
    public TensorData(float[] arr) {
        if (arr == null) {
            throw new IllegalArgumentException(
                "\nArray nulo."
            );
        }

        this.data = Arrays.copyOf(arr, arr.length);
        this.offset = 0;
        this.len = arr.length;
    }

    /**
     * Inicializa um {@code TensorData} a partir um tamanho especificado.
     * <p>
     *      O conteúdo estará zerado.
     * </p>
     * @param len tamanho desejado.
     */
    public TensorData(int len) {
        if (len < 1) {
            throw new IllegalArgumentException(
                "\nTamanho deve ser maior que zero, mas recebido " + len + "."
            );
        }

        this.data = new float[len];
        this.offset = 0;
        this.len = data.length;
    }

    /**
     * Retorna um novo {@code TensorData} que é uma visualização a partir
     * da instância local.
     * @param start índicide inicial do conjunto de elementos.
     * @param len tamamnho do conjunto de elementos.
     * @return view da instância local.
     */
    public TensorData view(int start, int len) {
        if (start < 0 || len < 0 || start + len > this.len) {
            throw new IllegalArgumentException("\nView fora do limite do tensor data.");
        }
        return new TensorData(data, offset + start, len);
    }

    /**
     * Retorna o valor contido no conjunto de dados.
     * @param i índice linear baseado no array de dados.
     * @return valor obtido.
     */
    public float get(int i) {
        return data[offset + i];
    }

    /**
     * Altera o valor contido no conjunto de dados.
     * @param x valor base.
     * @param i índice linear baseado no array de dados.
     * @return TensorData local alterado.
     */
    public TensorData set(float x, int i) {
        data[offset + i] = x;
        return this;
    }

    /**
     * Preenche todo o conjunto de dados.
     * @param x valor base.
     * @return TensorData local alterado.
     */
    public TensorData fill(float x) {
        final float[] d = data;
        final int inicio = offset;
        final int fim = offset + len;

        for (int i = inicio; i < fim; i++) {
            d[i] = x;
        }
        
        return this;
    }

	/**
	 * Preenche o conteúdo de dados usando um contador iniciado com
	 * valor 1 que é alterado a cada elemento.
	 * @param pos contador crescente (1, 2, 3, ...), caso falso o
	 * contador é decrescente (-1, -2, -3, ...).
     * @return TensorData local alterado.
	 */
    public TensorData fillCount(boolean pos) {
        final float[] d = data;
        final int inicio = offset;
        final int fim = offset + len;

        if (pos) {
            for (int i = inicio; i < fim; i++) {
                d[i] = i - inicio + 1;
            }
        } else {
            for (int i = inicio; i < fim; i++) {
                d[i] = -(i - inicio + 1);
            }
        }

        return this;
    }

    /**
     * Zera os valores do conjunto de dados.
     * @return TensorData local alterado.
     */
    public TensorData zero() {
        return fill(0.0f);
    }

    /**
     * Copia o conteúdo do array para o conjunto de dados.
     * @param arr {@code array} base.
     * @return TensorData local alterado.
     */
    public TensorData copy(float[] arr) {
        final int n = len;
        if (arr.length != n) {
            throw new IllegalArgumentException(
                "\nTamanho do array (" + arr.length + ") deve ser igual ao tamanho de dados (" + n + ")."
            );
        }

        System.arraycopy(arr, 0, data, offset, n);

        return this;
    }

    /**
     * Copia o conteúdo de outro conjunto de dados para a isntância local.
     * @param td {@code TensorData} base.
     * @return TensorData local alterado.
     */
    public TensorData copy(TensorData td) {
        final int n = len();

        if (td.len() != n) {
            throw new IllegalArgumentException(
                "\nTamanhos incompatíveis: " + n + " != " + td.len()
            );
        }

        System.arraycopy(td.data, td.offset, this.data, this.offset, n);

        return this;
    }

    /**
     * Calcula o produto interno entre a instância local e o TensorData
     * fornecido.
     * Equivalente ao {@code ddot} do BLAS
     * @see <a href="https://www.netlib.org/lapack/explore-html/d1/dcc/group__dot_ga2a42ecc597403b22ad786715c739196b.html">
     *      LAPACK ddot</a>
     * @param td {@code TensorData} base.
     * @return valor resultante do produto interno.
     */
    public float dot(TensorData td) {
        final int n = len();
        if (td.len() != n) {
            throw new IllegalArgumentException(
                "\nTamanhos incompatíveis."
            );
        }

        final float[] da = data;
        final float[] db = td.data;
        final int offA = offset;
        final int offB = td.offset;

        float soma = 0.0f;

        for (int i = 0; i < n; i++) {
            soma += da[offA + i] * db[offB + i];
        }

        return soma;        
    }

    /**
     * Realiza a operação {@code A + B*alfa}, onde:
     * <pre>
     *A = Instância Local
     *B = Outro TensorData
     *Alfa = multiplicador para os elementos de B
     * </pre>
     * Equivalente ao {@code axpy} do BLAS:
     * @see <a href="https://www.netlib.org/lapack/explore-html/d5/d4b/group__axpy.html">
     *      LAPACK axpy</a>
     * @param td {@code TensorData} base.
     * @param alpha escalar multiplicativo.
     * @return TensorData local alterado.
     */
    public TensorData add(TensorData td, float alpha) {
        final int n = len();
        if (td.len() != n) {
            throw new IllegalArgumentException(
                "\nTamanhos incompatíveis."
            );
        }

        final float[] da = data;
        final float[] db = td.data;
        final int offA = offset;
        final int offB = td.offset;

        for (int i = 0; i < n; i++) {
            da[offA + i] += alpha * db[offB + i];        
        }

        return this;
    }

    /**
     * Realiza a operação {@code A + B}, onde:
     * <pre>
     *A = Instância Local
     *B = Outro TensorData
     * </pre>
     * @param td {@code TensorData} base.
     * @return TensorData local alterado.
     */
    public TensorData add(TensorData td) {
        return add(td, 1.0f);
    }

    /**
     * Realiza a operação {@code A - B}, onde:
     * <pre>
     *A = Instância Local
     *B = Outro TensorData
     * </pre>
     * @param td {@code TensorData} base.
     * @return TensorData local alterado.
     */
    public TensorData sub(TensorData td) {
        final int n = len();
        if (n != td.len()) {
            throw new IllegalArgumentException(
                "\nAmbos os TensorData devem possuir o mesmo tamanho."
            );
        }

        final float[] a = this.data;
        final float[] b = td.data;
        final int baseA = this.offset;
        final int baseB = td.offset;

        for (int i = 0; i < n; i++) {
            a[baseA + i] -= b[baseB + i];
        }

        return this;
    }

    /**
     * Realiza a operação {@code A * B}, onde:
     * <pre>
     *A = Instância Local
     *B = Outro TensorData
     * </pre>
     * @param td {@code TensorData} base.
     * @return TensorData local alterado.
     */
    public TensorData mul(TensorData td) {
        final int n = len();
        if (n != td.len()) {
            throw new IllegalArgumentException(
                "\nAmbos os TensorData devem possuir o mesmo tamanho."
            );
        }

        final float[] a = this.data;
        final float[] b = td.data;
        final int baseA = this.offset;
        final int baseB = td.offset;

        for (int i = 0; i < n; i++) {
            a[baseA + i] *= b[baseB + i];
        }

        return this;
    }

    /**
     * Realiza a operação {@code A / B}, onde:
     * <pre>
     *A = Instância Local
     *B = Outro TensorData
     * </pre>
     * @param td {@code TensorData} base.
     * @return TensorData local alterado.
     */
    public TensorData div(TensorData td) {
        final int n = len();
        if (n != td.len()) {
            throw new IllegalArgumentException(
                "\nAmbos os TensorData devem possuir o mesmo tamanho."
            );
        }

        final float[] a = this.data;
        final float[] b = td.data;
        final int baseA = this.offset;
        final int baseB = td.offset;

        for (int i = 0; i < n; i++) {
            a[baseA + i] /= b[baseB + i];
        }

        return this;
    }

    /**
     * Adiciona um valor em todo o conjunto de dados.
     * @param x valor base.
     * @return TensorData local alterado.
     */
    public TensorData add(float x) {
        final float[] d = data;
        final int inicio = offset;
        final int fim = offset + len;

        for (int i = inicio; i < fim; i++) {
            d[i] += x;
        }

        return this;
    }
    
    /**
     * Subtrai um valor em todo o conjunto de dados.
     * @param x valor base.
     * @return TensorData local alterado.
     */
    public TensorData sub(float x) {
        return add(-x);
    }
 
    /**
     * Multiplica um valor em todo o conjunto de dados.
     * @param x valor base.
     * @return TensorData local alterado.
     */
    public TensorData mul(float x) {
        final float[] d = data;
        final int inicio = offset;
        final int fim = offset + len;

        for (int i = inicio; i < fim; i++) {
            d[i] *= x;
        }

        return this;
    }
    
    /**
     * Divide um valor em todo o conjunto de dados.
     * @param x valor base.
     * @return TensorData local alterado.
     */
    public TensorData div(float x) {
        final float[] d = data;
        final int inicio = offset;
        final int fim = offset + len;

        for (int i = inicio; i < fim; i++) {
            d[i] /= x;
        }

        return this;
    }

    /**
     * Adiciona um valor de acordo com um índice linear.
     * @param x valor base.
     * @param i índice baseado no array do conjunto de elementos.
     * @return TensorData local alterado.
     */
    public TensorData add(float x, int i) {
        if (i < 0 || i >= len) {
            throw new IllegalArgumentException(
                "\nÍndice " + i + " inválido."
            );
        }

        data[offset + i] += x;

        return this;
    }

    /**
     * Subtrai um valor de acordo com um índice linear.
     * @param x valor base.
     * @param i índice baseado no array do conjunto de elementos.
     * @return TensorData local alterado.
     */
    public TensorData sub(float x, int i) {
        return add(-x, i);
    }

    /**
     * Multiplica um valor de acordo com um índice linear.
     * @param x valor base.
     * @param i índice baseado no array do conjunto de elementos.
     * @return TensorData local alterado.
     */
    public TensorData mul(float x, int i) {
        if (i < 0 || i >= len) {
            throw new IllegalArgumentException(
                "\nÍndice " + i + " inválido."
            );
        }

        data[offset + i] *= x;

        return this;
    }

    /**
     * Divide um valor de acordo com um índice linear.
     * @param x valor base.
     * @param i índice baseado no array do conjunto de elementos.
     * @return TensorData local alterado.
     */
    public TensorData div(float x, int i) {
        if (i < 0 || i >= len) {
            throw new IllegalArgumentException(
                "\nÍndice " + i + " inválido."
            );
        }

        data[offset + i] /= x;

        return this;
    }

    /**
     * Adiciona o conteúdo do array ao conjunto de dados.
     * @param arr {@code array} base.
     * @return TensorData local alterado.
     */
    public TensorData add(float[] arr) {
        final int n = len();
        if (arr.length != n) {
            throw new IllegalArgumentException(
                "\nTamanho do array (" + arr.length + ") deve ser igual ao tamanho de dados (" + n + ")."
            );
        }

        final float[] d = data;

        for (int i = 0; i < n; i++) {
            d[offset + i] += arr[i];
        }

        return this;
    }

    /**
     * Subtrai o conteúdo do array ao conjunto de dados.
     * @param arr {@code array} base.
     * @return TensorData local alterado.
     */
    public TensorData sub(float[] arr) {
        final int n = len();
        if (arr.length != n) {
            throw new IllegalArgumentException(
                "\nTamanho do array (" + arr.length + ") deve ser igual ao tamanho de dados (" + n + ")."
            );
        }

        final float[] d = data;

        for (int i = 0; i < n; i++) {
            d[offset + i] -= arr[i];
        }

        return this;
    }
    
    /**
     * Multiplica o conteúdo do array ao conjunto de dados.
     * @param arr {@code array} base.
     * @return TensorData local alterado.
     */
    public TensorData mul(float[] arr) {
        final int n = len();
        if (arr.length != n) {
            throw new IllegalArgumentException(
                "\nTamanho do array (" + arr.length + ") deve ser igual ao tamanho de dados (" + n + ")."
            );
        }

        final float[] d = data;

        for (int i = 0; i < n; i++) {
            d[offset + i] *= arr[i];
        }

        return this;
    }
    
    /**
     * Divide o conteúdo do array ao conjunto de dados.
     * @param arr {@code array} base.
     * @return TensorData local alterado.
     */
    public TensorData div(float[] arr) {
        final int n = len();
        if (arr.length != n) {
            throw new IllegalArgumentException(
                "\nTamanho do array (" + arr.length + ") deve ser igual ao tamanho de dados (" + n + ")."
            );
        }

        final float[] d = data;

        for (int i = 0; i < n; i++) {
            d[offset + i] /= arr[i];
        }
    
        return this;
    }

    /**
     * Aplica uma função em todos os elementos do conjunto de dados.
     * @param fun função base.
     * @return TensorData local alterado.
     */
    public TensorData apply(FloatUnaryOperator fun) {
        final float[] d = data;
        final int inicio = offset;
        final int fim = inicio + len;

        for (int i = inicio; i < fim; i++) {
            d[i] = fun.apply(d[i]);
        }

        return this;
    }

    /**
     * Aplica uma função em todos os elementos do conjunto de dados base,
     * armazenando o resultado neste conjunto local.
     *
     * @param td  conjunto de dados base.
     * @param fun função a ser aplicada em cada elemento de {@code td}.
     * @return {@code TensorData} local alterado.
     */
    public TensorData apply(TensorData td, FloatUnaryOperator fun) {
        final int n = len();
        
        if (td.len() != n) {
            throw new IllegalArgumentException("\nTamanhos incompatíveis.");
        }

        final float[] da = data;
        final float[] db = td.data;
        final int offA = offset;
        final int offB = td.offset;

        for (int i = 0; i < n; i++) {
            da[offA + i] = fun.apply(db[offB + i]);
        }

        return this;
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
     * @param a {@code TensorData} A.
     * @param b {@code TensorData} B.
     * @param alpha {@code valor} escalar multiplicativo.
     * @return TensorData local alterado.
     */
    public TensorData addcmul(TensorData a, TensorData b, float alpha) {
        final int n = len();

        if (a.len() != n || b.len() != n) {
            throw new IllegalArgumentException("\nTamanhos incompatíveis.");
        }

        final float[] d  = data;
        final float[] db = a.data;
        final float[] dc = b.data;
        final int offA = offset;
        final int offB = a.offset;
        final int offC = b.offset;

        for (int i = 0; i < n; i++) {
            d[offA + i] += alpha * (db[offB + i] * dc[offC + i]);
        }

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
     * @param a {@code TensorData} numerador.
     * @param b {@code TensorData} denominador.
     * @param alpha {@code valor} escalar multiplicativo.
     * @return TensorData local alterado.
     */
    public TensorData addcdiv(TensorData a, TensorData b, float alpha) {
        final int n = len();
        if (a.len() != n || b.len() != n) {
            throw new IllegalArgumentException("\nTamanhos incompatíveis.");
        }

        final float[] d  = data;
        final float[] db = a.data;
        final float[] dc = b.data;
        final int offA = offset;
        final int offB = a.offset;
        final int offC = b.offset;

        for (int i = 0; i < n; i++) {
            d[offA + i] += alpha * (db[offB + i] / dc[offC + i]);
        }

        return this;
    }

    /**
     * Realiza a comparação elemento a elemento entre a instância local e
     * o conjunto de dados fornecido.
     * @param td {@code TensorData} para comparação.
     * @return TensorData local alterado.
     */
    public TensorData maxBetween(TensorData td) {
        final int n = len();
        if (td.len() != n)
            throw new IllegalArgumentException("\nTamanhos incompatíveis.");

        final float[] da = data;
        final float[] db = td.data;
        final int offA = offset;
        final int offB = td.offset;

        for (int i = 0; i < n; i++) {
            float v = db[offB + i];
            if (v > da[offA + i]) da[offA + i] = v;
        }

        return this;
    }

    /**
     * Aplica a função signum em todos os elementos do conjunto de dados.
     * @return TensorData local alterado.
     */
    public TensorData signum() {
        final float[] d = data;
        final int inicio = offset;
        final int fim = inicio + len;

        for (int i = inicio; i < fim; i++) {
            d[i] = Math.signum(d[i]);
        }

        return this;
    }

    /**
     * Calcula a raiz quadrada de cada elemento do conjunto de dados.
     * @return TensorData local alterado.
     */
    public TensorData sqrt() {
        final float[] d = data;
        final int inicio = offset;
        final int fim = inicio + len;

        for (int i = inicio; i < fim; i++) {
            d[i] = (float) Math.sqrt(d[i]);
        }

        return this;
    }

    /**
     * Retorna a soma dos elementos do conjunto de dados.
     * @return soma dos elementos.
     */
    public float sum() {
        final float[] d = data;
        final int inicio = offset;
        final int fim = inicio + len;
        
        float soma = 0.0f;
        for (int i = inicio; i < fim; i++) {
            soma += d[i];
        }

        return soma;      
    }

    /**
     * Retorna o valor máximo contido no conjunto de dados.
     * @return valor máximo.
     */
    public float max() {
        float max = get(0);
        final float[] d = data;
        final int inicio = offset + 1;
        final int fim = offset + len();
        
        for (int i = inicio; i < fim; i++) {
            if (d[i] > max) max = d[i];
        }

        return max;
    }

    /**
     * Retorna o valor mínimo contido no conjunto de dados.
     * @return valor mínimo.
     */
    public float min() {
        float min = get(0);
        final float[] d = data;
        final int inicio = offset + 1;
        final int fim = offset + len();
        
        for (int i = inicio; i < fim; i++) {
            if (d[i] < min) min = d[i];
        }

        return min;
    }

    /**
     * Retorna a média aritmética dos valores do conjunto de dados.
     * @return média.
     */
    public float mean() {
        return sum() / len;
    }

    /**
     * Retorna o desvio padrão dos valores do conjunto de dados.
     * @return desvio padrão.
     */
    public float std() {
        float media = mean();
        final float[] d = data;
        final int inicio = offset;
        final int fim = inicio + len();
        
        double soma = 0.0d;
        for (int i = inicio; i < fim; i++) {
            soma += Math.pow(d[i] - media, 2);
        }

        return (float) Math.sqrt(soma / len);
    }

	/**
	 * Restringe o conteúdo de dados entre um valor mínimo e máximo.
	 * @param min valor mínimo.
	 * @param max valor máximo.
     * @return TensorData local alterado.
	 */
    public TensorData clamp(float min, float max) {
        if (min >= max) {
            throw new IllegalArgumentException(
                "\nValor mínimo não pode ser maior ou igual ao valor máximo."
            );
        }

        final float[] d = data;
        final int inicio = offset;
        final int fim = inicio + len;

        for (int i = inicio; i < fim; i++) {
            d[i] = Math.clamp(d[i], min, max);
        }

        return this;
    }

    /**
     * Retorna o array do conjunto de dados.
     * <p>
     *      Essa ação retorna o conjunto completo de dados,
     *      assim views podem retornar o conjunto de dados original
     *      da qual foram criadas.
     * </p>
     * @return referência do conjunto de dados.
     */
    public float[] data() {
        return data;
    }

    /**
     * Retorna uma cópia do conjunto de dados.
     * <p>
     *      Essa ação retorna uma cópia dos dados internos, assim
     *      views retornam apenas uma cópia do seu próprio conteúdo
     *      local.
     * </p>
     * @return {@code clone} do conjunto de dados.
     */
    public float[] toArray() {
        return Arrays.copyOfRange(data, offset, offset + len);
    }

    /**
     * Retorna o tamanho do conjunto de dados.
     * <p>
     *      O resultado de tam() pode variar se o TensorData
     *      for uma view de outro conjunto de dados.
     * </p>
     * @return tamanho do conjunto de dados.
     */
    public int len() {
        return len;
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

        long tamArr = 4 * len();// float 4 bytes
        long tamOffset = 4;// int 4 bytes
        long tamTam = 4;// int 4 bytes
		return tamObj + tamArr + tamOffset + tamTam;
    }

    @Override
    public TensorData clone() {
        float[] novo = new float[len];
        System.arraycopy(data, offset, novo, 0, len);
        
        return new TensorData(novo);
    }

    /**
     * Verifica se o conjunto de dados é uma view de outra instância.
     * @return {@code true} se os dados forem uma view, {@code false} 
     * caso contrário.
     */
    public boolean isView() {
        return offset > 0 || len < data.length;
    }

    /**
     * Gera uma String personalizada representando o TensorData.
     * @return {@code String} formatada
     */
    private String buildInfo() {
        StringBuilder sb = new StringBuilder();

        // usando get() para adaptar quando for uma view
        
        String tab = "    ";
        sb.append("TensorData = [\n");
        sb.append(tab).append("[").append(get(0));
        
        if (len < 10) {
            for (int i = 1; i < len; i++) sb.append(", ").append(get(i));
        } else {
            // primeiros 5 elementos
            for (int i = 1; i < 5; i++) sb.append(", ").append(get(i));
            // ultimos 5 elementos
            sb.append(" ... "). append(get(len-5));
            for (int i = len-4; i < len; i++) sb.append(", ").append(get(i));
        }

        sb.append("]\n");
        sb.append(tab).append("View: ").append(isView()).append("\n");
        sb.append(tab).append("Tam: ").append(len).append("\n");
        sb.append("]");

        return sb.toString();
    }

    @Override
    public String toString() {
        return buildInfo();
    }

    /**
     * Exibe, via terminal, as informações do conjunto de dados.
     */
    public void print() {
        System.out.println(buildInfo());
    }
}
