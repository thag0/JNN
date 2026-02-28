package jnn.dataloader;

import jnn.core.tensor.Tensor;

/**
 * Conteiner para armazenar um conjunto de amostras.
 */
public class LoteAmostra {
    
    /**
     * Dados de entrada (X).
     */
    private Tensor[] xs;

    /**
     * Dados de saída (Y).
     */
    private Tensor[] ys;

    /**
     * Inicialzia um lote de amostras.
     * @param x Dados de entrada (X).
     * @param y Dados de saída (Y).
     */
    public LoteAmostra(Tensor[] x, Tensor[] y) {
        if (x.length != y.length) {
            throw new IllegalArgumentException(
                "\nQuantidade de elementos de X e Y devem ser iguais."
            );
        }

        this.xs = x;
        this.ys = y;
    }

    /**
     * Retorna os dados de entrada (X) do lote.
     * @return Dados de entrada (X) do lote.
     */
    public Tensor[] arrX() {
        return xs;
    }

    /**
     * Retorna os dados de saída (Y) do lote.
     * @return Dados de saída (Y) do lote.
     */
    public Tensor[] arrY() {
        return ys;
    }

    /**
    * Retorna o número de amostras no lote.
    * @return Número de amostras no lote.
    */
    public int tam() {
        return xs.length;
    }

}
