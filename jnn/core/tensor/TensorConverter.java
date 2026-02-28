package jnn.core.tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Conversor de dados em Tensor.
 * @see jnn.core.tensor.Tensor
 */
public class TensorConverter {

    /**
     * Construtor privado.
     */
    private TensorConverter() {}
    
    /**
     * Converte um objeto em um {@code Tensor}.
     * @param obj objeto base, deve ser um array de elementos que possa 
     * ser convertido para o formato {@code float}.
     * @return um {@code Tensor} representado pelo array.
     */
    public static Tensor tensor(Object obj) {
		if (obj instanceof Tensor) {
			return (Tensor) obj;
		}

        Class<?> cls = obj.getClass();

        if (!cls.isArray()) {
            throw new IllegalArgumentException(
                "\nObjeto deve ser um array, recebido \"" + cls.getSimpleName() + "\"."
            );
        }

        int[] shape = getShape(obj);
        validarShape(obj, shape, 0);
        float[] dados = achatarDados(obj, shape);

		return new Tensor(dados.length)
        .copiar(dados)
        .reshape(shape);
    }

    /**
     * Obtem o shape do array.
     * @param obj objeto base, deve ser um array.
     * @return shape.
     */
    private static int[] getShape(Object obj) {
        int prof = 0;
        Class<?> cls = obj.getClass();
        while (cls.isArray()) {
            prof++;
            cls = cls.getComponentType();
        }

        int[] shape = new int[prof];
        Object atual = obj;

        for (int i = 0; i < prof; i++) {
            int tam = java.lang.reflect.Array.getLength(atual);
            shape[i] = tam;

            if (tam > 0) {
                atual = java.lang.reflect.Array.get(atual, 0);
            } else {
                break;
            }
        }

        return shape;
    }
    
    /**
     * Verifica se o shape é válido para usar em um tensor.
     * @param obj objeto base.
     * @param shape shape previamente calculado.
     * @param dim dimensão de checagem.
     */
    private static void validarShape(Object obj, int[] shape, int dim) {
        if (dim == shape.length) {
            if (obj instanceof Number) return;
            throw new IllegalArgumentException("\nProfundidade inconsistente no array.");
        }

        if (obj == null || !obj.getClass().isArray()) {
            throw new IllegalArgumentException("\nEstrutura irregular no nível " + dim);
        }

        int len = java.lang.reflect.Array.getLength(obj);

        if (len != shape[dim]) {
            throw new IllegalArgumentException(
                "\nArray irregular na dimensão " + dim +
                ". Esperado: " + shape[dim] +
                ", encontrado: " + len
            );
        }

        for (int i = 0; i < len; i++) {
            Object elem = java.lang.reflect.Array.get(obj, i);
            validarShape(elem, shape, dim + 1);
        }
    }


    /**
     * Transforma os dados do objeto em um array linear.
     * @param obj objeto base, deve ser um array.
     * @param shape formato do array.
     * @return dados achatados, convertidos em {@code double} (usado pelo Tensor).
     */
    private static float[] achatarDados(Object obj, int[] shape) {
        List<Float> list = new ArrayList<>();
        achatarRecursivo(obj, list);
        
        float[] arr = new float[list.size()];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = list.get(i);
        }
        
        return arr;
    }

    /**
     * Percorre as dimensões do objeto, achatando seu dados.
     * @param obj objeto base, deve ser um array.
     * @param list lista de dados dos novos elementos achatados.
     */
    private static void achatarRecursivo(Object obj, List<Float> list) {
        if (obj == null) return;

        Class<?> cls = obj.getClass();
        if (!cls.isArray()) {
            if (obj instanceof Number) {
                list.add(((Number) obj).floatValue());
            } else {
                throw new IllegalArgumentException("Valor não numérico: " + obj);
            }
            return;
        }

        int tam = java.lang.reflect.Array.getLength(obj);
        for (int i = 0; i < tam; i++) {
            Object elem = java.lang.reflect.Array.get(obj, i);
            achatarRecursivo(elem, list);
        }
    }

}
