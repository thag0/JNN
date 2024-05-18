package jnn.core.tensor;

/**
 * <h2>
 *      Variável primitiva de tensores
 * </h2>
 * <p>
 *      Essa é a unidade mais básica de variável usada pela biblioteca.
 * </p>
 * Implementei isso para poder compartilhar referências de
 * variáveis entre tensores, principalmente usando slicing.
 * <p>
 *      Uma variável é basicamente um valor numérico encapsulado. Ela
 *      pode fazer operações aritméticas simples (soma, subtração, 
 *      multiplicação, divisão) além de operações essenciais como get e set.
 * </p>
 * <p>
 *      As operações não possuem retorno e a alteração do conteúdo é feita localmente.
 * </p>
 */
public class Variavel implements Cloneable {
    
    /**
     * Real valor contido na variável.
     */
    private double valor;

    /**
     * Inicializa uma variável com valor igual a zero.
     */
    public Variavel() {
        valor = 0.0;
    }

    /**
     * Inicializa uma variável com o valor desejado.
     * @param x valor desejado
     */
    public Variavel(double x) {
        valor = x;
    }

    /**
     * Inicializa uma variável a partir do valor de 
     * outra variável.
     * <p>
     *      Apenas o valor da variável informada será copiado.
     * </p>
     * @param v variável desejada
     */
    public Variavel(Variavel v) {
        valor = v.valor;
    }

    /**
     * Adiciona o valor informado no conteúdo da variável.
     * @param x valor desejado.
     */
    public void add(double x) {
        valor += x;
    }

    /**
     * Adiciona o valor local usando o valor da variável informada.
     * @param v variável desejada.
     */
    public void add(Variavel v) {
        valor += v.valor;
    }

    /**
     * Subtrai o valor informado no conteúdo da variável.
     * @param x valor desejado.
     */
    public void sub(double x) {
        valor -= x;
    }

    /**
     * Subtrai o valor local usando o valor da variável informada.
     * @param v variável desejada.
     */
    public void sub(Variavel v) {
        valor -= v.valor;
    }

    /**
     * Multiplica o valor informado no conteúdo da variável.
     * @param x valor desejado.
     */
    public void mult(double x) {
        valor *= x;
    }

    /**
     * Multiplica o valor local usando o valor da variável informada.
     * @param v variável desejada.
     */
    public void mult(Variavel v) {
        valor *= v.valor;
    }

    /**
     * Divide o valor informado no conteúdo da variável.
     * @param x valor desejado.
     */
    public void div(double x) {
        valor /= x;
    }
   
    /**
     * Divide o valor local usando o valor da variável informada.
     * @param v variável desejada.
     */
    public void div(Variavel v) {
        valor /= v.valor;
    }

    /**
     * Atribui o valor informado no conteúdo da variável.
     * @param x valor desejado.
     */
    public void set(double x) {
        valor = x;
    }

     /**
     * Atribui o valor local usando o valor da variável informada.
     * @param v variável desejada.
     */
    public void set(Variavel v) {
        valor = v.valor;
    }

    /**
     * Retorna o valor numérico contido na variável.
     * @return valor da variável.
     */
    public double get() {
        return valor;
    }

    // métodos especiais

    /**
     * Adiciona localmente o resultado da soma entre variáveis recebidas.
     * @param v1 {@code Variavel} 1
     * @param v2 {@code Variavel} 2
     */
    public void addSoma(Variavel v1, Variavel v2) {
        valor += (v1.valor + v2.valor);
    }

    /**
     * Adiciona localmente o resultado da diferença entre variáveis recebidas.
     * @param v1 {@code Variavel} 1
     * @param v2 {@code Variavel} 2
     */
    public void addSub(Variavel v1, Variavel v2) {
        valor += (v1.valor - v2.valor);
    }

    /**
     * Adiciona localmente o resultado do produto entre variáveis recebidas.
     * @param v1 {@code Variavel} 1
     * @param v2 {@code Variavel} 2
     */
    public void addMult(Variavel v1, Variavel v2) {
        valor += (v1.valor * v2.valor);
    }

    /**
     * Adiciona localmente o resultado da divisão entre variáveis recebidas.
     * @param v1 {@code Variavel} 1
     * @param v2 {@code Variavel} 2
     */
    public void addDiv(Variavel v1, Variavel v2) {
        valor += (v1.valor / v2.valor);
    }

    /**
     * Subtrai localmente o resultado da soma entre variáveis recebidas.
     * @param v1 {@code Variavel} 1
     * @param v2 {@code Variavel} 2
     */
    public void subSoma(Variavel v1, Variavel v2) {
        valor -= (v1.valor + v2.valor);
    }

    /**
     * Subtrai localmente o resultado da diferença entre variáveis recebidas.
     * @param v1 {@code Variavel} 1
     * @param v2 {@code Variavel} 2
     */
    public void subSub(Variavel v1, Variavel v2) {
        valor -= (v1.valor - v2.valor);
    }

    /**
     * Subtrai localmente o resultado do produto entre variáveis recebidas.
     * @param v1 {@code Variavel} 1
     * @param v2 {@code Variavel} 2
     */
    public void subMult(Variavel v1, Variavel v2) {
        valor -= (v1.valor * v2.valor);
    }

    /**
     * Subtrai localmente o resultado da divisão entre variáveis recebidas.
     * @param v1 {@code Variavel} 1
     * @param v2 {@code Variavel} 2
     */
    public void subDiv(Variavel v1, Variavel v2) {
        valor -= (v1.valor / v2.valor);
    }

    /**
     * Multiplica localmente o resultado da soma entre variáveis recebidas.
     * @param v1 {@code Variavel} 1
     * @param v2 {@code Variavel} 2
     */
    public void multSoma(Variavel v1, Variavel v2) {
        valor *= (v1.valor + v2.valor);
    }

    /**
     * Multiplica localmente o resultado da diferneça entre variáveis recebidas.
     * @param v1 {@code Variavel} 1
     * @param v2 {@code Variavel} 2
     */
    public void multSub(Variavel v1, Variavel v2) {
        valor *= (v1.valor - v2.valor);
    }

    /**
     * Multiplica localmente o resultado do produto entre variáveis recebidas.
     * @param v1 {@code Variavel} 1
     * @param v2 {@code Variavel} 2
     */
    public void multMult(Variavel v1, Variavel v2) {
        valor *= (v1.valor * v2.valor);
    }

    /**
     * Multiplica localmente o resultado da divisão entre variáveis recebidas.
     * @param v1 {@code Variavel} 1
     * @param v2 {@code Variavel} 2
     */
    public void multDiv(Variavel v1, Variavel v2) {
        valor *= (v1.valor / v2.valor);
    }

    /**
     * Divide localmente o resultado da soma entre variáveis recebidas.
     * @param v1 {@code Variavel} 1
     * @param v2 {@code Variavel} 2
     */
    public void divSoma(Variavel v1, Variavel v2) {
        valor /= (v1.valor + v2.valor);
    }

    /**
     * Divide localmente o resultado da diferneça entre variáveis recebidas.
     * @param v1 {@code Variavel} 1
     * @param v2 {@code Variavel} 2
     */
    public void divSub(Variavel v1, Variavel v2) {
        valor /= (v1.valor - v2.valor);
    }

    /**
     * Divide localmente o resultado do produto entre variáveis recebidas.
     * @param v1 {@code Variavel} 1
     * @param v2 {@code Variavel} 2
     */
    public void divMult(Variavel v1, Variavel v2) {
        valor /= (v1.valor * v2.valor);
    }

    /**
     * Divide localmente o resultado da divisão entre variáveis recebidas.
     * @param v1 {@code Variavel} 1
     * @param v2 {@code Variavel} 2
     */
    public void divtDiv(Variavel v1, Variavel v2) {
        valor /= (v1.valor / v2.valor);
    }

    @Override
    public boolean equals(Object obj) {
        return (obj instanceof Variavel) && (this.valor == ((Variavel) obj).valor);
    }

    @Override
    public Variavel clone() {
        return new Variavel(valor);
    }

    @Override
    public String toString() {
        return "[Variavel: " + valor + "]";
    }
}
