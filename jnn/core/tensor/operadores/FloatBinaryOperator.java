package jnn.core.tensor.operadores;

@FunctionalInterface
public interface FloatBinaryOperator {
    
    public float apply(float x, float y);

}