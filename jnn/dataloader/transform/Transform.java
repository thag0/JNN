package jnn.dataloader.transform;

import jnn.core.tensor.Tensor;

/**
 * Inferface para aplicar transformações em {@code Tensor} dentro
 * do {@code DataLoader}
 * @see jnn.core.tensor.Tensor Tensor
 * @see jnn.dataloader.DataLoader DataLoader
 */
@FunctionalInterface
public interface Transform {
    
    /**
     * Aplica uma transformação no {@code Tensor}.
     * @param in {@code Tensor} base.
     * @param out {@code Tensor} de destino.
     */
    void apply(Tensor in, Tensor out);

    /**
     * Verifica se os shapes dos tensores de entrada e saída são compatíveis.
     * @param in {@code Tensor} de entrada.
     * @param out {@code Tensor} de destino.
     */
    default public void validarShapes(Tensor in, Tensor out) {
        if (!in.compShape(out)) {
            throw new IllegalArgumentException(
                "\nOs tensores de entrada e saída devem conter o mesmo tamanho, mas " +
                "in = " + in.shapeStr() + " e out = " + out.shapeStr()
            );
        }
    }

    /**
     * Retorna o nome da transformação.
     * @return Nome da transformação.
     */
    default public String nome() {
        return getClass().getSimpleName();
    }

}
