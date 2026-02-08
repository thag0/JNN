package jnn.core.tensor.operadores;

/**
 * Interface simples para funções com dois operadors float.
 */
@FunctionalInterface
public interface FloatBinaryOperator {
    
    /**
     * Aplica a função sobre os valores dados.
     * @param x primeiro valor.
     * @param y segundo valor.
     * @return valor produzido pela função;
     */
    public float apply(float x, float y);

}