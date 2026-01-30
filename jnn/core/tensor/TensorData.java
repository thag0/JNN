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
 * @see {@link jnn.core.tensor.Tensor}
 */
public class TensorData {

    /**
     * Array de elementos.
     */
    private final float[] dados;

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
    private final int tam; 

    /**
     * Inicializa um {@code TensorData} a partir de dados bem definidos.
     * <p>
     *      O conteúdo é referenciado como o mesmo do array.
     * </p>
     * @param arr {@code array} base.
     * @param offset indice inicial a partir do array base.
     * @param tam tamanho final do conjunto de elementos.
     */
    public TensorData(float[] arr, int offset, int tam) {
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
    public TensorData(float[] arr) {
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

        this.dados = new float[tam];
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
        if (inicio < 0 || tam < 0 || inicio + tam > this.tam) {
            throw new IllegalArgumentException("\nView fora do limite do tensor data.");
        }
        return new TensorData(dados, offset + inicio, tam);
    }

    /**
     * Retorna o valor contido no conjunto de dados.
     * @param id índice linear baseado no array de dados.
     * @return valor obtido.
     */
    public float get(int id) {
        return dados[offset + id];
    }

    /**
     * Altera o valor contido no conjunto de dados.
     * @param x valor base.
     * @param id índice linear baseado no array de dados.
     * @return TensorData local alterado.
     */
    public TensorData set(Number x, int id) {
        dados[offset + id] = x.floatValue();
        return this;
    }

    /**
     * Preenche todo o conjunto de dados.
     * @param x valor base.
     * @return TensorData local alterado.
     */
    public TensorData preencher(Number x) {
        final float val = x.floatValue();
        
        for (int i = 0; i < tam; i++) {
            dados[offset + i] = val;
        }
        
        return this;
    }

    public TensorData preencherContador(boolean cres) {
        final int tam = tam();

        if (cres) {
            for (int i = 0; i < tam; i++) {
                dados[offset + i] = i+1;
            }
        } else {
            for (int i = 0; i < tam; i++) {
                dados[offset + i] = - i -1;
            }
        }

        return this;
    }

    /**
     * Zera os valores do conjunto de dados.
     * @return TensorData local alterado.
     */
    public TensorData zero() {
        return preencher(0);
    }

    /**
     * Copia o conteúdo do array para o conjunto de dados.
     * @param arr {@code array} base.
     * @return TensorData local alterado.
     */
    public TensorData copiar(float[] arr) {
        final int n = tam;
        if (arr.length != n) {
            throw new IllegalArgumentException(
                "\nTamanho do array (" + arr.length + ") deve ser igual ao tamanho de dados (" + n + ")."
            );
        }

        System.arraycopy(arr, 0, dados, offset, n);

        return this;
    }

    /**
     * Copia o conteúdo de outro conjunto de dados para a isntância local.
     * @param td {@code TensorData} base.
     * @return TensorData local alterado.
     */
    public TensorData copiar(TensorData td) {
        final int n = tam();

        if (td.tam() != n) {
            throw new IllegalArgumentException(
                "\nTamanhos incompatíveis: " + n + " != " + td.tam()
            );
        }

        System.arraycopy(td.dados, td.offset, this.dados, this.offset, n);

        return this;
    }

    /**
     * Calcula o produto interno entre a instância local e o TensorData
     * fornecido.
     * Equivalente ao {@code ddot} do BLAS {@link https://www.netlib.org/lapack/explore-html/d0/d8c/group__dot.html}
     * @param td {@code TensorData} base.
     * @return valor resultante do produto interno.
     */
    public float dot(TensorData td) {
        final int n = tam();
        if (td.tam() != n) {
            throw new IllegalArgumentException(
                "\nTamanhos incompatíveis."
            );
        }

        final float[] da = dados;
        final float[] db = td.dados;
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
     * Equivalente ao {@code axpy} do BLAS {@link https://www.netlib.org/lapack/explore-html/d5/d4b/group__axpy.html}
     * @param td {@code TensorData} base.
     * @return TensorData local alterado.
     */
    public TensorData add(TensorData td, float alfa) {
        final int n = tam();
        if (td.tam() != n) {
            throw new IllegalArgumentException(
                "\nTamanhos incompatíveis."
            );
        }

        final float[] da = dados;
        final float[] db = td.dados;
        final int offA = offset;
        final int offB = td.offset;

        for (int i = 0; i < n; i++) {
            da[offA + i] += alfa * db[offB + i];        
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
        final int n = tam();
        if (n != td.tam()) {
            throw new IllegalArgumentException(
                "\nAmbos os TensorData devem possuir o mesmo tamanho."
            );
        }

        final float[] a = this.dados;
        final float[] b = td.dados;
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
        final int n = tam();
        if (n != td.tam()) {
            throw new IllegalArgumentException(
                "\nAmbos os TensorData devem possuir o mesmo tamanho."
            );
        }

        final float[] a = this.dados;
        final float[] b = td.dados;
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
        final int n = tam();
        if (n != td.tam()) {
            throw new IllegalArgumentException(
                "\nAmbos os TensorData devem possuir o mesmo tamanho."
            );
        }

        final float[] a = this.dados;
        final float[] b = td.dados;
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
        final int n = tam();
        for (int i = 0; i < n; i++) {
            dados[offset + i] += x;
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
        final int n = tam();
        for (int i = 0; i < n; i++) {
            dados[offset + i] *= x;
        }

        return this;
    }
    
    /**
     * Divide um valor em todo o conjunto de dados.
     * @param x valor base.
     * @return TensorData local alterado.
     */
    public TensorData div(float x) {
        final int n = tam();
        for (int i = 0; i < n; i++) {
            dados[offset + i] /= x;
        }

        return this;
    }

    /**
     * Adiciona um valor de acordo com um índice linear.
     * @param x valor base.
     * @param id índice baseado no array do conjunto de elementos.
     * @return TensorData local alterado.
     */
    public TensorData add(float x, int id) {
        if (id < 0 || id >= tam) {
            throw new IllegalArgumentException(
                "\nÍndice " + id + " inválido."
            );
        }

        dados[offset + id] += x;

        return this;
    }

    /**
     * Subtrai um valor de acordo com um índice linear.
     * @param x valor base.
     * @param id índice baseado no array do conjunto de elementos.
     * @return TensorData local alterado.
     */
    public TensorData sub(float x, int id) {
        return add(-x, id);
    }

    /**
     * Multiplica um valor de acordo com um índice linear.
     * @param x valor base.
     * @param id índice baseado no array do conjunto de elementos.
     * @return TensorData local alterado.
     */
    public TensorData mul(float x, int id) {
        if (id < 0 || id >= tam) {
            throw new IllegalArgumentException(
                "\nÍndice " + id + " inválido."
            );
        }

        dados[offset + id] *= x;

        return this;
    }

    /**
     * Divide um valor de acordo com um índice linear.
     * @param x valor base.
     * @param id índice baseado no array do conjunto de elementos.
     * @return TensorData local alterado.
     */
    public TensorData div(float x, int id) {
        if (id < 0 || id >= tam) {
            throw new IllegalArgumentException(
                "\nÍndice " + id + " inválido."
            );
        }

        dados[offset + id] /= x;

        return this;
    }

    /**
     * Adiciona o conteúdo do array ao conjunto de dados.
     * @param arr {@code array} base.
     * @return TensorData local alterado.
     */
    public TensorData add(float[] arr) {
        final int n = tam();
        if (arr.length != n) {
            throw new IllegalArgumentException(
                "\nTamanho do array (" + arr.length + ") deve ser igual ao tamanho de dados (" + n + ")."
            );
        }

        for (int i = 0; i < n; i++) {
            dados[offset + i] += arr[i];
        }

        return this;
    }

    /**
     * Subtrai o conteúdo do array ao conjunto de dados.
     * @param arr {@code array} base.
     * @return TensorData local alterado.
     */
    public TensorData sub(float[] arr) {
        final int n = tam();
        if (arr.length != n) {
            throw new IllegalArgumentException(
                "\nTamanho do array (" + arr.length + ") deve ser igual ao tamanho de dados (" + n + ")."
            );
        }

        for (int i = 0; i < n; i++) {
            dados[offset + i] -= arr[i];
        }

        return this;
    }
    
    /**
     * Multiplica o conteúdo do array ao conjunto de dados.
     * @param arr {@code array} base.
     * @return TensorData local alterado.
     */
    public TensorData mul(float[] arr) {
        final int n = tam();
        if (arr.length != n) {
            throw new IllegalArgumentException(
                "\nTamanho do array (" + arr.length + ") deve ser igual ao tamanho de dados (" + n + ")."
            );
        }

        for (int i = 0; i < n; i++) {
            dados[offset + i] *= arr[i];
        }

        return this;
    }
    
    /**
     * Divide o conteúdo do array ao conjunto de dados.
     * @param arr {@code array} base.
     * @return TensorData local alterado.
     */
    public TensorData div(float[] arr) {
        final int n = tam();
        if (arr.length != n) {
            throw new IllegalArgumentException(
                "\nTamanho do array (" + arr.length + ") deve ser igual ao tamanho de dados (" + n + ")."
            );
        }

        for (int i = 0; i < n; i++) {
            dados[offset + i] /= arr[i];
        }
    
        return this;
    }

    /**
     * Aplica uma função em todos os elementos do conjunto de dados.
     * @param fun função base.
     * @return TensorData local alterado.
     */
    public TensorData aplicar(FloatUnaryOperator fun) {
        final int inicio = offset;
        final int fim = inicio + tam();

        for (int i = inicio; i < fim; i++) {
            dados[i] = fun.apply(dados[i]);
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
    public TensorData aplicar(TensorData td, FloatUnaryOperator fun) {
        final int n = tam();
        
        if (td.tam() != n) {
            throw new IllegalArgumentException("\nTamanhos incompatíveis entre TensorData.");
        }

        final float[] da = dados;
        final float[] db = td.dados;
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
     * {@link https://docs.pytorch.org/docs/stable/generated/torch.addcmul.html}
     * @param a {@code TensorData} A.
     * @param b {@code TensorData} B.
     * @param alfa {@code valor} escalar multiplicativo.
     * @return TensorData local alterado.
     */
    public TensorData addcmul(TensorData a, TensorData b, float alfa) {
        final int n = tam();

        if (a.tam() != n || b.tam() != n) {
            throw new IllegalArgumentException("\nTamanhos incompatíveis.");
        }

        final float[] d  = dados;
        final float[] db = a.dados;
        final float[] dc = b.dados;
        final int offA = offset;
        final int offB = a.offset;
        final int offC = b.offset;

        for (int i = 0; i < n; i++) {
            d[offA + i] += alfa * (db[offB + i] * dc[offC + i]);
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
     * {@link https://docs.pytorch.org/docs/stable/generated/torch.addcdiv.html}
     * @param a {@code TensorData} numerador.
     * @param b {@code TensorData} denominador.
     * @param alfa {@code valor} escalar multiplicativo.
     * @return TensorData local alterado.
     */
    public TensorData addcdiv(TensorData a, TensorData b, float alfa) {
        final int n = tam();
        if (a.tam() != n || b.tam() != n) {
            throw new IllegalArgumentException("\nTamanhos incompatíveis.");
        }

        final float[] d  = dados;
        final float[] db = a.dados;
        final float[] dc = b.dados;
        final int offA = offset;
        final int offB = a.offset;
        final int offC = b.offset;

        for (int i = 0; i < n; i++) {
            d[offA + i] += alfa * (db[offB + i] / dc[offC + i]);
        }

        return this;
    }

    /**
     * Realiza a comparação elemento a elemento entre a instância local e
     * o conjunto de dados fornecido.
     * @param td {@code TensorData} para comparação.
     * @return TensorData local alterado.
     */
    public TensorData maxEntre(TensorData td) {
        final int n = tam();
        if (td.tam() != n)
            throw new IllegalArgumentException("\nTamanhos incompatíveis.");

        final float[] da = dados;
        final float[] db = td.dados;
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
        final int inicio = offset;
        final int fim = inicio + tam();

        for (int i = inicio; i < fim; i++) {
            dados[i] = Math.signum(dados[i]);
        }

        return this;
    }

    /**
     * Calcula a raiz quadrada de cada elemento do conjunto de dados.
     * @return TensorData local alterado.
     */
    public TensorData sqrt() {
        final int inicio = offset;
        final int fim = inicio + tam();

        for (int i = inicio; i < fim; i++) {
            dados[i] = (float) Math.sqrt(dados[i]);
        }

        return this;
    }

    /**
     * Retorna a soma dos elementos do conjunto de dados.
     * @return soma dos elementos.
     */
    public float soma() {
        float s = 0.0f;
        final int inicio = offset;
        final int fim = inicio + tam();

        for (int i = inicio; i < fim; i++) {
            s += dados[i];
        }

        return s;      
    }

    /**
     * Retorna o valor máximo contido no conjunto de dados.
     * @return valor máximo.
     */
    public float max() {
        float max = get(0);
        final int inicio = offset;
        final int fim = inicio + tam();
        
        for (int i = inicio; i < fim; i++) {
            if (dados[i] > max) max = dados[i];
        }

        return max;
    }

    /**
     * Retorna o valor mínimo contido no conjunto de dados.
     * @return valor mínimo.
     */
    public float min() {
        float min = get(0);
        final int inicio = offset;
        final int fim = inicio + tam();

        for (int i = inicio; i < fim; i++) {
            if (dados[i] < min) min = dados[i];
        }

        return min;
    }

    /**
     * Retorna a média aritmética dos valores do conjunto de dados.
     * @return média.
     */
    public float media() {
        return soma() / tam();
    }

    /**
     * Retorna o desvio padrão dos valores do conjunto de dados.
     * @return desvio padrão.
     */
    public float desvp() {
        float media = media();
        float soma = 0.0f;
        final int inicio = offset;
        final int fim = inicio + tam();

        for (int i = inicio; i < fim; i++) {
            soma += Math.pow(dados[i] - media, 2);
        }

        return (float) Math.sqrt(soma / tam());
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

        final int inicio = offset;
        final int fim = inicio + tam();

        for (int i = inicio; i < fim; i++) {
            dados[i] = Math.clamp(dados[i], min, max);
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
        return dados;
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
    public float[] paraArray() {
        return Arrays.copyOfRange(dados, offset, offset + tam);
    }

    /**
     * Retorna o tamanho do conjunto de dados.
     * <p>
     *      O resultado de tam() pode variar se o TensorData
     *      for uma view de outro conjunto de dados.
     * </p>
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

        long tamArr = 4 * tam();// float 8 bytes
        long tamOffset = 4;// int 4 bytes
        long tamTam = 4;// int 4 bytes
		return tamObj + tamArr + tamOffset + tamTam;
    }

    @Override
    public TensorData clone() {
        float[] novo = Arrays.copyOfRange(this.dados, this.offset, this.offset + this.tam);
        return new TensorData(novo, 0, novo.length);
    }

    /**
     * Verifica se o conjunto de dados é uma view de outra instância.
     * @return {@code true} se os dados forem uma view, {@code false} 
     * caso contrário.
     */
    public boolean isView() {
        return offset > 0 || tam < dados.length;
    }

    /**
     * Gera uma String personalizada representando o TensorData.
     * @return {@code String} formatada
     */
    private String construirInfo() {
        StringBuilder sb = new StringBuilder();

        // usando get() para adaptar quando for uma view
        
        String tab = "    ";
        sb.append("TensorData = [\n");
        sb.append(tab).append("[").append(get(0));
        
        if (tam < 10) {
            for (int i = 1; i < tam; i++) sb.append(", ").append(get(i));
        } else {
            // primeiros 5 elementos
            for (int i = 1; i < 5; i++) sb.append(", ").append(get(i));
            // ultimos 5 elementos
            sb.append(" ... "). append(get(tam-5));
            for (int i = tam-4; i < tam; i++) sb.append(", ").append(get(i));
        }

        sb.append("]\n");
        sb.append(tab).append("View: ").append(isView()).append("\n");
        sb.append(tab).append("Tam: ").append(tam).append("\n");
        sb.append("]");

        return sb.toString();
    }

    @Override
    public String toString() {
        return construirInfo();
    }

    /**
     * Exibe, via terminal, as informações do conjunto de dados.
     */
    public void print() {
        System.out.println(construirInfo());
    }
}
