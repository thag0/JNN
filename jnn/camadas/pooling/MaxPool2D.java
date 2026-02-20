package jnn.camadas.pooling;

/**
 * <h2>
 *    Camada de agrupamento máximo
 * </h2>
 * <p>
 *    A camada de agrupamento máximo é um componente utilizado para reduzir a 
 *    dimensionalidade espacial dos dados, preservando as características mais 
 *    importantes para a saída.
 * </p>
 * <p>
 *    Durante a operação de agrupamento máximo, a entrada é dividida em regiões 
 *    menores usando uma márcara e o valor máximo de cada região é salvo. 
 *    Essencialmente, a camada realiza a operação de subamostragem, mantendo apenas 
 *    as informações mais relevantes.
 * </p>
 * Exemplo simples de operação Max Pooling para uma região 2x2 com máscara 2x2:
 * <pre>
 *entrada = [
 *    [[1, 2],
 *     [3, 4]]
 *]
 * 
 *saida = [
 *    [4]
 *]
 * </pre>
 * <p>
 *    A camada de max pooling não possui parâmetros treináveis nem função de ativação.
 * </p>
 */
public class MaxPool2D extends Pool2DBase implements Cloneable {

	/**
	 * Instancia uma nova camada de max pooling, definindo o formato do
	 * filtro que será aplicado em cada entrada da camada.
	 * <p>
	 *    O formato do filtro deve conter as dimensões da entrada da
	 *    camada (altura, largura).
	 * </p>
	 * <p>
	 *    Por padrão, os valores de strides serão os mesmo usados para
	 *    as dimensões do filtro, exemplo:
	 * </p>
	 * <pre>
	 *filtro = (2, 2)
	 *stride = (2, 2) // valor padrão
	 * </pre>
	 * @param filtro formato do filtro de max pooling.
	 * @throws IllegalArgumentException se o formato do filtro não atender as
	 * requisições.
	 */
	public MaxPool2D(int[] filtro) {
		this(filtro, filtro);// filtro == stride
	}

	/**
	 * Instancia uma nova camada de max pooling, definindo o formato do filtro 
	 * e os strides (passos) que serão aplicados em cada entrada da camada.
	 * <p>
	 *    O formato do filtro e dos strides devem conter as dimensões da entrada 
	 *    da camada (altura, largura).
	 * </p>
	 * @param filtro formato do filtro de max pooling.
	 * @param stride strides que serão aplicados ao filtro.
	 * @throws IllegalArgumentException se o formato do filtro não atender as
	 * requisições.
	 * @throws IllegalArgumentException se os strides não atenderem as requisições.
	 */
	public MaxPool2D(int[] filtro, int[] stride) {
		super(filtro, stride);
	}

	/**
	 * Instancia uma nova camada de max pooling, definindo o formato do filtro, 
	 * formato de entrada e os strides (passos) que serão aplicados em cada entrada 
	 * da camada.
	 * <p>
	 *    O formato do filtro e dos strides devem conter as dimensões da entrada 
	 *    da camada (altura, largura).
	 * </p>
	 * A camada será automaticamente construída usando o formato de entrada especificado.
	 * @param entrada formato de entrada para a camada.
	 * @param filtro formato do filtro de max pooling.
	 * @param stride strides que serão aplicados ao filtro.
	 * @throws IllegalArgumentException se o formato do filtro não atender as
	 * requisições.
	 * @throws IllegalArgumentException se os strides não atenderem as requisições.
	 */
	public MaxPool2D(int[] entrada, int[] filtro, int[] stride) {
		this(filtro, stride);
		construir(entrada);
	}

	/**
	 * Constroi a camada MaxPooling, inicializando seus atributos.
	 * <p>
	 *    O formato de entrada da camada deve seguir o padrão:
	 * </p>
	 * <pre>
	 *    formEntrada = (profundidade, altura, largura)
	 * </pre>
	 */
	@Override
	public void construir(int[] shape) {
		super.construir(shape);
		modo = "max";
	}

	@Override
	public MaxPool2D clone() {
		return (MaxPool2D) super.clone();
	}

}
