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
     * @param v {@code Variavel} base.
     */
    public Variavel(Variavel v) {
        valor = v.valor;
    }

    /**
     * Realiza a operação
     * <pre>V = V + X</pre>
     * @param x valor base.
	 * @return {@code Variavel} local alterada.
     */
    public Variavel add(Number x) {
        valor += x.doubleValue();
        return this;
    }

    /**
     * Realiza a operação
     * <pre>V = V + X</pre>
     * @param x {@code Variavel} base.
	 * @return {@code Variavel} local alterada.
     */
    public Variavel add(Variavel x) {
        valor += x.valor;
        return this;
    }

    /**
     * Realiza a operação
     * <pre>V = V - X</pre>
     * @param x valor base.
	 * @return {@code Variavel} local alterada.
     */
    public Variavel sub(Number x) {
        valor -= x.doubleValue();
        return this;
    }

    /**
     * Realiza a operação
     * <pre>V = V - X</pre>
     * @param x {@code Variavel} base.
	 * @return {@code Variavel} local alterada.
     */
    public Variavel sub(Variavel x) {
        valor -= x.valor;
        return this;
    }

    /**
     * Realiza a operação
     * <pre>V = V * X</pre>
     * @param x valor base.
	 * @return {@code Variavel} local alterada.
     */
    public Variavel mul(Number x) {
        valor *= x.doubleValue();
        return this;
    }

    /**
     * Realiza a operação
     * <pre>V = V * X</pre>
     * @param x {@code Variavel} base.
	 * @return {@code Variavel} local alterada.
     */
    public Variavel mul(Variavel x) {
        valor *= x.valor;
        return this;
    }

    /**
     * Realiza a operação
     * <pre>V = V / X</pre>
     * @param x valor base.
	 * @return {@code Variavel} local alterada.
     */
    public Variavel div(Number x) {
        valor /= x.doubleValue();
        return this;
    }
   
    /**
     * Realiza a operação
     * <pre>V = V / X</pre>
     * @param x {@code Variavel} base.
	 * @return {@code Variavel} local alterada.
     */
    public Variavel div(Variavel x) {
        valor /= x.valor;
        return this;
    }

    /**
     * Realiza a operação
     * <pre>V = X</pre>
     * @param x valor base.
	 * @return {@code Variavel} local alterada.
     */
    public Variavel set(Number x) {
        valor = x.doubleValue();
        return this;
    }

    /**
     * Realiza a operação
     * <pre>V = X</pre>
     * @param x {@code Variavel} base.
	 * @return {@code Variavel} local alterada.
     */
    public Variavel set(Variavel x) {
        valor = x.valor;
        return this;
    }

    /**
     * Retorna o valor numérico contido na variável.
     * @return valor da variável.
     */
    public double get() {
        return valor;
    }

    /**
     * Realiza a operação
     * <pre>V = 0</pre>
	 * @return {@code Variavel} local alterada.
     */
    public Variavel zero() {
        valor = 0.0;
        return this;
    }

    // métodos especiais

    /**
     * Realiza a operação
     * <pre>V = V + (X + Y)</pre>
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
	 * @return {@code Variavel} local alterada.
     */
    public Variavel addadd(Variavel x, Variavel y) {
        valor += (x.valor + y.valor);
        return this;
    }

    /**
     * Realiza a operação
     * <pre>V = V + (X - Y)</pre>
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
	 * @return {@code Variavel} local alterada.
     */
    public Variavel addsub(Variavel x, Variavel y) {
        valor += (x.valor - y.valor);
        return this;
    }

    /**
     * Realiza a operação
     * <pre>V = V + (X * Y)</pre>
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
	 * @return {@code Variavel} local alterada.
     */
    public Variavel addmul(Variavel x, Variavel y) {
        valor += (x.valor * y.valor);
        return this;
    }

    /**
     * Realiza a operação
     * <pre>V = V + (X / Y)</pre>
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
	 * @return {@code Variavel} local alterada.
     */
    public Variavel adddiv(Variavel x, Variavel y) {
        valor += (x.valor / y.valor);
        return this;
    }

    /**
     * Realiza a operação
     * <pre>V = V - (X + Y)</pre>
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
	 * @return {@code Variavel} local alterada.
     */
    public Variavel subadd(Variavel x, Variavel y) {
        valor -= (x.valor + y.valor);
        return this;
    }

    /**
     * Realiza a operação
     * <pre>V = V - (X - Y)</pre>
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
	 * @return {@code Variavel} local alterada.
     */
    public Variavel subsub(Variavel x, Variavel y) {
        valor -= (x.valor - y.valor);
        return this;
    }

    /**
     * Realiza a operação
     * <pre>V = V - (X * Y)</pre>
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
	 * @return {@code Variavel} local alterada.
     */
    public Variavel submul(Variavel x, Variavel y) {
        valor -= (x.valor * y.valor);
        return this;
    }

    /**
     * Realiza a operação
     * <pre>V = V - (X / Y)</pre>
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
	 * @return {@code Variavel} local alterada.
     */
    public Variavel subdiv(Variavel x, Variavel y) {
        valor -= (x.valor / y.valor);
        return this;
    }

    /**
     * Realiza a operação
     * <pre>V = V * (X + Y)</pre>
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
	 * @return {@code Variavel} local alterada.
     */
    public Variavel muladd(Variavel x, Variavel y) {
        valor *= (x.valor + y.valor);
        return this;
    }

    /**
     * Realiza a operação
     * <pre>V = V * (X - Y)</pre>
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
	 * @return {@code Variavel} local alterada.
     */
    public Variavel mulsub(Variavel x, Variavel y) {
        valor *= (x.valor - y.valor);
        return this;
    }

    /**
     * Realiza a operação
     * <pre>V = V * (X * Y)</pre>
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
	 * @return {@code Variavel} local alterada.
     */
    public Variavel mulmul(Variavel x, Variavel y) {
        valor *= (x.valor * y.valor);
        return this;
    }

    /**
     * Realiza a operação
     * <pre>V = V * (X / Y)</pre>
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
	 * @return {@code Variavel} local alterada.
     */
    public Variavel muldiv(Variavel x, Variavel y) {
        valor *= (x.valor / y.valor);
        return this;
    }

    /**
     * Realiza a operação
     * <pre>V = V / (X + Y)</pre>
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
	 * @return {@code Variavel} local alterada.
     */
    public Variavel divadd(Variavel x, Variavel y) {
        valor /= (x.valor + y.valor);
        return this;
    }

    /**
     * Realiza a operação
     * <pre>V = V / (X - Y)</pre>
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
	 * @return {@code Variavel} local alterada.
     */
    public Variavel divsub(Variavel x, Variavel y) {
        valor /= (x.valor - y.valor);
        return this;
    }

    /**
     * Realiza a operação
     * <pre>V = V / (X * Y)</pre>
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
	 * @return {@code Variavel} local alterada.
     */
    public Variavel divmul(Variavel x, Variavel y) {
        valor /= (x.valor * y.valor);
        return this;
    }

    /**
     * Realiza a operação
     * <pre>V = V / (X / Y)</pre>
     * @param x {@code Variavel} 1
     * @param y {@code Variavel} 2
	 * @return {@code Variavel} local alterada.
     */
    public Variavel divdiv(Variavel x, Variavel y) {
        valor /= (x.valor / y.valor);
        return this;
    }

    /**
     * Realiza a operação
     * <pre>V > X</pre>
     * @param x {@code Variavel} desejada.
     * @return resultado da verificação.
     */
    public boolean maior(Variavel x) {
        return valor > x.valor;
    }

    /**
     * Realiza a operação
     * <pre>V > X</pre>
     * @param x {@code Variavel} desejada.
     * @return resultado da verificação.
     */
    public boolean maior(Number x) {
        return valor > x.doubleValue();
    }

    /**
     * Realiza a operação
     * <pre>V < X</pre>
     * @param x {@code Variavel} desejada.
     * @return resultado da verificação.
     */
    public boolean menor(Variavel x) {
        return valor < x.valor;
    }

    /**
     * Realiza a operação
     * <pre>V < X</pre>
     * @param x {@code Variavel} desejada.
     * @return resultado da verificação.
     */
    public boolean menor(Number x) {
        return valor < x.doubleValue();
    }

    /**
     * Exibe, {@code via terminal}, o conteúdo da variável.
     */
    public void print() {
        System.out.println(toString());
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
        return "[Var: " + valor + "]";
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
