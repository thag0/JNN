package jnn.camadas.pooling;

/**
 * <h1>
 *    Camada de agrupamento médio
 * </h1>
 * <p>
 *    A camada de agrupamento médio é um componente utilizado para reduzir a 
 *    dimensionalidade espacial dos dados, preservando as características mais 
 *    importantes para a saída.
 * </p>
 * <p>
 *    Durante a operação de agrupamento médio, a entrada é dividida em regiões 
 *    menores usando uma máscara e a média de cada região é calculada e salva. 
 *    Essencialmente, a camada realiza a operação de subamostragem, calculando a 
 *    média das informações em cada região.
 * </p>
 * Exemplo simples de operação Avg Pooling para uma região 2x2 com máscara 2x2:
 * <pre>
 *entrada = [
 *    [[1, 2],
 *     [3, 4]]
 *]
 * 
 *saida = [2.5]
 * </pre>
 * <p>
 *    A camada de avg pooling não possui parâmetros treináveis nem função de ativação.
 * </p>
 */
public class AvgPool2D extends Pool2DBase {

	/**
	 * Instancia uma nova camada de average pooling, definindo o formato do filtro 
	 * e os strides (passos) que serão aplicados em cada entrada da camada.
	 * <p>
	 *    O formato do filtro e dos strides devem conter as dimensões da entrada 
	 *    da camada (altura, largura).
	 * </p>
	 * @param filtro formato do filtro de average pooling.
	 * @param stride strides que serão aplicados ao filtro.
	 */
	public AvgPool2D(int[] filtro, int[] stride) {
		super(filtro, stride);
	}

	/**
	 * Instancia uma nova camada de average pooling, definindo o formato do
	 * filtro que será aplicado em cada entrada da camada.
	 * <p>
	 *    O formato do filtro deve conter as dimensões da entrada da
	 *    camada (altura, largura).
	 * </p>
	 * <p>
	 *    Por padrão, os valores de strides serão os mesmos usados para
	 *    as dimensões do filtro, exemplo:
	 * </p>
	 * <pre>
	 *filtro = (2, 2)
	 *stride = (2, 2) // valor padrão
	 * </pre>
	 * @param filtro formato do filtro de average pooling.
	 */
	public AvgPool2D(int[] filtro) {
		super(filtro, filtro);// filtro == stride
	}

	/**
	 * Instancia uma nova camada de average pooling, definindo o formato do filtro, 
	 * formato de entrada e os strides (passos) que serão aplicados em cada entrada 
	 * da camada.
	 * <p>
	 *    O formato do filtro e dos strides devem conter as dimensões da entrada 
	 *    da camada (altura, largura).
	 * </p>
	 * A camada será automaticamente construída usando o formato de entrada especificado.
	 * @param entrada formato de entrada para a camada.
	 * @param filtro formato do filtro de average pooling.
	 * @param stride strides que serão aplicados ao filtro.
	 */
	public AvgPool2D(int[] entrada, int[] filtro, int[] stride) {
		this(filtro, stride);
		construir(entrada);
	}

	/**
	 * Constroi a camada AvgPooling, inicializando seus atributos.
	 * <p>
	 *    O formato de entrada da camada deve seguir o padrão:
	 * </p>
	 * <pre>
	 *    formEntrada = (canais, altura, largura)
	 * </pre>
	 */
	@Override
	public void construir(int[] shape) {
		super.construir(shape);
		modo = "avg";
	}

	@Override
	public AvgPool2D clone() {
		return (AvgPool2D) super.clone();
	}

}
