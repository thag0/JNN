package jnn.core.tensor.operadores;

/**
 * Interface simples para funções com um único operador float.
 */
@FunctionalInterface
public interface FloatUnaryOperator {
    
    /**
     * Aplica a função sobre o valor dado.
     * @param x valor de entrada.
     * @return valor produzido pela função;
     */
    public float apply(float x);

}