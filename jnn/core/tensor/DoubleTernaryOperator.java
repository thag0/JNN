package jnn.core.tensor;

@FunctionalInterface
public interface DoubleTernaryOperator {
    double applyAsDouble(double a, double b, double c);
}
