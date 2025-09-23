package jnn.core.tensor;

/**
 * Representa uma operação com três operadores do tipo {@code double}.
 */
@FunctionalInterface
public interface DoubleTernaryOperator {

    /**
     * Aplica o operador nos operandos recebidos.
     * @param a primeiro operador.
     * @param b segundo operador.
     * @param c terceiro operador.
     * @return resultado da operação.
     */
    double applyAsDouble(double a, double b, double c);

}