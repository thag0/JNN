package jnn.modelos;

import java.util.Iterator;

import jnn.camadas.Camada;

/**
 * Iterator para o modelo Sequencial.
 */
class ModelIterator implements Iterator<Camada> {
    
    /**
     * Índice atual da camada.
     */
    int id = 0;
    
    /**
     * Array de elementos.
     */
    Camada[] arr;

    /**
     * Tamanho total do array de elementos.
     */
    int tam;

    /**
     * Inicializa um ModelIterator.
     * @param cs array de {@code Camada}.
     */
    public ModelIterator(Camada[] cs) {
        arr = cs;
        tam = cs.length;
    }

    @Override
    public boolean hasNext() {
        return id < tam;
    }

    @Override
    public Camada next() {
        return arr[id++];
    }
}
