package jnn.models;

import java.util.Iterator;

import jnn.layers.Layer;

/**
 * Iterator para o modelo Sequencial.
 */
class ModelIterator implements Iterator<Layer> {
    
    /**
     * Índice atual da camada.
     */
    int id = 0;
    
    /**
     * Array de elementos.
     */
    Layer[] layers;

    /**
     * Tamanho total do array de elementos.
     */
    int len;

    /**
     * Inicializa um ModelIterator.
     * @param ls array de {@code Camada}.
     */
    public ModelIterator(Layer[] ls) {
        layers = ls;
        len = ls.length;
    }

    @Override
    public boolean hasNext() {
        return id < len;
    }

    @Override
    public Layer next() {
        return layers[id++];
    }
}
