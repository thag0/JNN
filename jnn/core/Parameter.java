package jnn.core;

import jnn.JNN;
import jnn.core.tensor.Tensor;

/**
 * Conteiner para pesos e bias.
 */
public class Parameter {

    /**
     * Peso.
     */
    public final Tensor weight;
    
    /**
     * Gradiente do peso.
     */
    public final Tensor grad;

    /**
     * Inicializa um novo parâmetro.
     * @param name nome para identificação.
     * @param shape formato desejado.
     */
    public Parameter(String name, int... shape) {
        weight = JNN.zeros(shape).name(name);
        grad = JNN.zeros(shape).name("grad " + name);
    }

    /**
     * Inicializa um novo parâmetro a partir de um tensor de base, os valores de peso
     * serão copiados.
     * @param name nome para identificação.
     * @param like {@code Tensor} base.
     */
    public Parameter(String name, Tensor like) {
        weight = like.clone();
        grad = JNN.zeros(like.shape()).name("grad " + name);
    }

    /**
     * Inicializa um novo parâmetro.
     * @param shape formato desejado.
     */
    public Parameter(int... shape) {
        this("Parameter", shape);
    }

    /**
     * Retorna o shape do peso.
     * @return shape do peso.
     */
    public int[] shape() {
        return weight.shape().clone();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        
        sb.append(name()).append(" (")

        .append('"').append(weight.name()).append('"')
        .append(", shape ").append(JNNutils.arrayStr(weight.shape()))
        .append(", len: ").append(JNNutils.formatSizeBytes(weight.sizeBytes()))
        
        .append(")");

        return sb.toString();
    }

    /**
     * Exibe as informações do parâmetro.
     */
    public void print() {
        System.out.println(toString());
    }

    @Override
    public Parameter clone() {
        Parameter clone = new Parameter(weight.name(), weight);
        clone.grad.copy(grad);
        return clone;
    }

    /**
     * Retorna o nome do parâmetro, que é o mesmo nome da classe.
     * @return nome do parâmetro.
     */
    public String name() {
        return getClass().getSimpleName();
    }

}