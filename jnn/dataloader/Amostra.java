package jnn.dataloader;

import jnn.core.tensor.Tensor;

/**
 * Conteiner mais simples de um conjunto de entrada (X) e
 * saída (Y) de um Dataset.
 */
public class Amostra {

    /**
     * Amostra de entrada.
     */
    private Tensor x;

    /**
     * Saída referente à amostra.
     */
    private Tensor y;

    /**
     * Nome da amostra, pra facilitar o debug.
     */
    private String nome = "Amostra";

    /**
     * Inicializa uma nova {@code Amostra}.
     * @param x {@code Tensor} com dados de entrada.
     * @param y {@code Tensor} com dados de saída.
     */
    public Amostra(Tensor x, Tensor y) {
        if (x == null || y == null) {
            throw new IllegalArgumentException(
                "\nDados X e Y da amostra não podem ser nulos."
            );
        }

        this.x = x;
        this.y = y;
    }

    /**
     * Inicializa uma nova {@code Amostra}.
     * @param x {@code Tensor} com dados de entrada.
     * @param y {@code Tensor} com dados de saída.
     * @param nome {@code String} para um nome personalizado da amostra.
     */
    public Amostra(Tensor x, Tensor y, String nome) {
        this(x, y);
    
        nome = nome.trim();
        if (!(nome.isEmpty()) || !(nome.isBlank())) {
            this.nome = nome;
        }
    }

    /**
     * Retorna o valor de entrada da amostra.
     * @return {@code Tensor} com dado X.
     */
    public Tensor x() {
        return x;
    }
    
    /**
     * Retorna o valor de saída da amostra.
     * @return {@code Tensor} com dado Y.
     */
    public Tensor y() {
        return y;
    }

    /**
     * Altera o valor do {@code Tensor} de X.
     * @param t novo {@code Tensor} com valor de X da amostra.
     */
    public void setX(Tensor t) {
        if (t != null) {
            this.x = t;
        }
    }

    /**
     * Altera o valor do {@code Tensor} de Y.
     * @param t novo {@code Tensor} com valor de Y da amostra.
     */
    public void setY(Tensor t) {
        if (t != null) {
            this.y = t;
        }
    }

    private String info() {
        StringBuilder sb = new StringBuilder();
        String spc = " ".repeat(4);

        sb.append(nome).append(" = [\n");
        sb.append(spc).append("X: ").append(x.shapeStr()).append("\n");
        sb.append(spc).append("Y: ").append(y.shapeStr()).append("\n");
        sb.append("]\n");
        return sb.toString();
    }

    @Override
    public String toString() {
        return info();
    }

    /**
     * Exibe, via terminal, um resumo da amostra.
     */
    public void print() {
        System.out.println(info());
    }
}
