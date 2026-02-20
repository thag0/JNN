package jnn.core.tensor.operadores;

/**
 * Interface simples para funções com três operadors float.
 */
@FunctionalInterface
public interface FloatTernaryOperator {
    
    /**
     * Aplica a função sobre os valores dados.
     * @param x primeiro valor.
     * @param y segundo valor.
     * @param z terceiro valor.
     * @return valor produzido pela função;
     */
    public float apply(float x, float y, float z);

}