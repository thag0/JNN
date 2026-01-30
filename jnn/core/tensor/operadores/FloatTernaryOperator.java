package jnn.core.tensor.operadores;

@FunctionalInterface
public interface FloatTernaryOperator {
    
    public float apply(float x, float y, float z);

}