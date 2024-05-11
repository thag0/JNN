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
 */
public class Variavel {
    
    /**
     * Real valor contido na variável.
     */
    private double valor;

    /**
     * Inicializa uma variável com valor igual a zero.
     */
    public Variavel() {
        valor = 0.0d;
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
        valor = v.get();
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
        valor += v.get();
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
        valor -= v.get();
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
        valor *= v.get();
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
        valor /= v.get();
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
        valor = v.get();
    }

    /**
     * Retorna o valor numérico contido na variável.
     * @return valor da variável.
     */
    public double get() {
        return valor;
    }

    @Override
    public Variavel clone() {
        return new Variavel(valor);
    }

    @Override
    public String toString() {
        return "(Var: " + valor + ")";
    }
}
