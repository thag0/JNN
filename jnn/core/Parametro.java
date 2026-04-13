package jnn.core;

import jnn.JNN;
import jnn.core.tensor.Tensor;

/**
 * Conteiner para pesos e bias.
 */
public class Parametro {

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
     * @param nome nome para identificação.
     * @param shape formato desejado.
     */
    public Parametro(String nome, int... shape) {
        weight = JNN.zeros(shape).nome(nome);
        grad = JNN.zeros(shape).nome("grad " + nome);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        
        sb.append("Parametro (")
        
        .append('"')
        .append(weight.nome())
        .append('"')
        .append(" = ")
        .append(JNNutils.arrayStr(weight.shape()))
        
        .append(", tam: ")
        .append(JNNutils.formatarTamBytes(weight.tamBytes()))
        
        .append(")");

        return sb.toString();
    }

    /**
     * Exibe as informações do parâmetro.
     */
    public void print() {
        System.out.println(toString());
    }

}