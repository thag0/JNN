package jnn.core.tensor;

/**
 * <h2>
 *      Variável primitiva de tensores
 * </h2>
 * <p>
 *      Essa é a unidade mais básica de variável usada pela biblioteca.
 * </p>
 *      Implementei isso para poder compartilhar referências de
 *      variáveis entre tensores, principalmente usando operação que
 *      alteram a visualização de tensores mas que devem referencias os
 *      mesmos objetos.
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
    public Variavel(Number x) {
        valor = x.doubleValue();
    }

    /**
     * Inicializa uma variável a partir do valor de 
     * outra variável.
     * <p>
     *      Apenas o valor da variável informada será copiado.
     * </p>
     * @param x variável desejada
     */
    public Variavel(Variavel x) {
        valor = x.valor;
    }

    /**
     * Adiciona o valor informado no conteúdo da variável.
     * @param x valor desejado.
     */
    public void add(Number x) {
        valor += x.doubleValue();
    }

    /**
     * Adiciona o valor local usando o valor da variável informada.
     * @param x variável desejada.
     */
    public void add(Variavel x) {
        valor += x.valor;
    }

    /**
     * Subtrai o valor informado no conteúdo da variável.
     * @param x valor desejado.
     */
    public void sub(Number x) {
        valor -= x.doubleValue();
    }

    /**
     * Subtrai o valor local usando o valor da variável informada.
     * @param x variável desejada.
     */
    public void sub(Variavel x) {
        valor -= x.valor;
    }

    /**
     * Multiplica o valor informado no conteúdo da variável.
     * @param x valor desejado.
     */
    public void mul(Number x) {
        valor *= x.doubleValue();
    }

    /**
     * Multiplica o valor local usando o valor da variável informada.
     * @param x variável desejada.
     */
    public void mul(Variavel x) {
        valor *= x.valor;
    }

    /**
     * Divide o valor informado no conteúdo da variável.
     * @param x valor desejado.
     */
    public void div(Number x) {
        valor /= x.doubleValue();
    }
   
    /**
     * Divide o valor local usando o valor da variável informada.
     * @param x variável desejada.
     */
    public void div(Variavel x) {
        valor /= x.valor;
    }

    /**
     * Atribui o valor informado no conteúdo da variável.
     * @param x valor desejado.
     */
    public void set(Number x) {
        valor = x.doubleValue();
    }

     /**
     * Atribui o valor local usando o valor da variável informada.
     * @param x variável desejada.
     */
    public void set(Variavel x) {
        valor = x.valor;
    }

    /**
     * Retorna o valor numérico contido na variável.
     * @return valor da variável.
     */
    public double get() {
        return valor;
    }

    /**
     * Zera o valor da variável.
     */
    public void zero() {
        valor = 0;
    }

    // métodos especiais

    /**
     * Adiciona localmente o resultado da soma entre variáveis recebidas.
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
     */
    public void addSoma(Variavel x, Variavel y) {
        valor += (x.valor + y.valor);
    }

    /**
     * Adiciona localmente o resultado da diferença entre variáveis recebidas.
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
     */
    public void addSub(Variavel x, Variavel y) {
        valor += (x.valor - y.valor);
    }

    /**
     * Adiciona localmente o resultado do produto entre variáveis recebidas.
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
     */
    public void addMul(Variavel x, Variavel y) {
        valor += (x.valor * y.valor);
    }

    /**
     * Adiciona localmente o resultado da divisão entre variáveis recebidas.
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
     */
    public void addDiv(Variavel x, Variavel y) {
        valor += (x.valor / y.valor);
    }

    /**
     * Subtrai localmente o resultado da soma entre variáveis recebidas.
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
     */
    public void subSoma(Variavel x, Variavel y) {
        valor -= (x.valor + y.valor);
    }

    /**
     * Subtrai localmente o resultado da diferença entre variáveis recebidas.
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
     */
    public void subSub(Variavel x, Variavel y) {
        valor -= (x.valor - y.valor);
    }

    /**
     * Subtrai localmente o resultado do produto entre variáveis recebidas.
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
     */
    public void subMul(Variavel x, Variavel y) {
        valor -= (x.valor * y.valor);
    }

    /**
     * Subtrai localmente o resultado da divisão entre variáveis recebidas.
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
     */
    public void subDiv(Variavel x, Variavel y) {
        valor -= (x.valor / y.valor);
    }

    /**
     * Multiplica localmente o resultado da soma entre variáveis recebidas.
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
     */
    public void mulSoma(Variavel x, Variavel y) {
        valor *= (x.valor + y.valor);
    }

    /**
     * Multiplica localmente o resultado da diferneça entre variáveis recebidas.
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
     */
    public void mulSub(Variavel x, Variavel y) {
        valor *= (x.valor - y.valor);
    }

    /**
     * Multiplica localmente o resultado do produto entre variáveis recebidas.
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
     */
    public void mulMul(Variavel x, Variavel y) {
        valor *= (x.valor * y.valor);
    }

    /**
     * Multiplica localmente o resultado da divisão entre variáveis recebidas.
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
     */
    public void mulDiv(Variavel x, Variavel y) {
        valor *= (x.valor / y.valor);
    }

    /**
     * Divide localmente o resultado da soma entre variáveis recebidas.
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
     */
    public void divSoma(Variavel x, Variavel y) {
        valor /= (x.valor + y.valor);
    }

    /**
     * Divide localmente o resultado da diferneça entre variáveis recebidas.
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
     */
    public void divSub(Variavel x, Variavel y) {
        valor /= (x.valor - y.valor);
    }

    /**
     * Divide localmente o resultado do produto entre variáveis recebidas.
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
     */
    public void divMul(Variavel x, Variavel y) {
        valor /= (x.valor * y.valor);
    }

    /**
     * Divide localmente o resultado da divisão entre variáveis recebidas.
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
     */
    public void divtDiv(Variavel x, Variavel y) {
        valor /= (x.valor / y.valor);
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

    /**
     * Calcula o tamanho em {@code bytes} da variavel, 
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

        return tamObj + 8; // +8 da variável double
    }
}
