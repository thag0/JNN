package jnn.core.tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Conversor de dados em Tensor.
 * @see jnn.core.tensor.Tensor
 */
public class TensorConverter {
    
    /**
     * Converte um objeto em um {@code Tensor}.
     * @param obj objeto base, deve ser um array de elementos que possa 
     * ser convertido para o formato {@code double}.
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
        double[] dados = achatarDados(obj, shape);

		return new Tensor(dados, shape);
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
     * Transforma os dados do objeto em um array linear.
     * @param obj objeto base, deve ser um array.
     * @param shape formato do array.
     * @return dados achatados, convertidos em {@code double} (usado pelo Tensor).
     */
    private static double[] achatarDados(Object obj, int[] shape) {
        List<Double> list = new ArrayList<>();
        achatarRecursivo(obj, list);
        
        double[] arr = new double[list.size()];
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
    private static void achatarRecursivo(Object obj, List<Double> list) {
        if (obj == null) return;

        Class<?> cls = obj.getClass();
        if (!cls.isArray()) {
            if (obj instanceof Number) {
                list.add(((Number) obj).doubleValue());
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
