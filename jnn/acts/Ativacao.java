package jnn.acts;

import jnn.camadas.Conv2D;
import jnn.camadas.Densa;
import jnn.core.tensor.Tensor;

/**
 * <h2>
 *		Ativação base
 * </h2>
 * <p>
 *		As funções de ativação são usadas para melhorar a capacidade de modelagem
 *		dos dados em que os modelos treinados estão sendo usados.
 * </p>
 * <p>
 *		Novas funções de ativações devem sobrescrever os métodos existentes 
 * 		{@code forward()} e {@code backward()}.
 * </p>
 */
public abstract class Ativacao {

	/**
	 * Construtor privado.
	 */
	protected Ativacao() {}

	/**
	 * Função usada para o forward pass.
	 * @param x valor de entrada.
	 * @return valor calculado pela ativação.
	 */
	protected float fx(float x) {
		throw new UnsupportedOperationException(
			"\nNão fx implementado para " + nome() + "."
		);
	}

	/**
	 * Função usada para o backward pass.
	 * @param x valor de entrada.
	 * @return valor calculado pela ativação.
	 */
	protected float dx(float x) {
		throw new UnsupportedOperationException(
			"\nNão dx implementado para " + nome() + "."
		);
	}

	/**
	 * Calcula o resultado da ativação de acordo com a função configurada.
	 * @param x {@code Tensor} de entrada.
	 * @param dest {@code Tensor} de destino.
	 */
	public void forward(Tensor x, Tensor dest) {
		final int offX = x.offset();
		final int offS = dest.offset();
		final float[] dataX = x.array();
		final float[] dataS = dest.array();

		final int tam = x.tam();

		for (int i = 0; i < tam; i++) {
			dataS[offX + i] = fx(dataX[offS + i]);
		}
	}

	/**
	 * Calcula o resultado da ativação de acordo com a função configurada.
	 * @param x {@code Tensor} de entrada.
	 * @return {@code Tensor} contendo resultado.
	 */
	public Tensor forward(Tensor x) {
		Tensor res = new Tensor(x.shape());
		forward(x, res);
		return res;
	}

	/**
	 * Calcula o resultado da derivada da função de ativação de acordo 
	 * com a função configurada
	 * @param x {@code Tensor} de entrada.
	 * @param g {@code Tensor} contendo os gradientes em relação a entrada.
	 * @param dest {@code Tensor} de destino.
	 */
	public void backward(Tensor x, Tensor g, Tensor dest) {
		// derivada da entrada * gradiente

		final int offX = x.offset();
		final int offG = g.offset();
		final int offS = dest.offset();
		final float[] dataX = x.array();
		final float[] dataG = g.array();
		final float[] dataS = dest.array();

		final int tam = x.tam();

		for (int i = 0; i < tam; i++) {
			dataS[offX + i] = dx(dataX[offS + i]) * dataG[offG + i];
		}
	}

	/**
	 * Calcula o resultado da derivada da função de ativação de acordo 
	 * com a função configurada
	 * @param x {@code Tensor} de entrada.
	 * @param g {@code Tensor} contendo os gradientes em relação a entrada.
	 * @return {@code Tensor} contendo resultado.
	 */
	public Tensor backward(Tensor x, Tensor g) {
		Tensor res = new Tensor(x.shape());
		backward(x, g, res);
		return res;
	}

	/**
	 * Implementação especifíca para camadas densas.
	 * <p>
	 *    Criada para dar suporte a ativações especiais.
	 * </p>
	 * @param camada camada convolucional.
	 * @param g {@code Tensor} contendo gradiente de saída da camada.
	 */
	public void backward(Densa camada, Tensor g) {
		//por padrão chamar o método da própria ativação
		backward(camada._buffer, g, camada._gradSaida);
	}

	/**
	 * Implementação especifíca para camadas convolucionais.
	 * <p>
	 *    Criada para dar suporte a ativações especiais.
	 * </p>
	 * @param camada camada convolucional.
	 * @param g {@code Tensor} contendo gradiente de saída da camada.
	 */
	public void backward(Conv2D camada, Tensor g) {
		//por padrão chamar o método da própria ativação
		backward(camada._buffer, g, camada._gradSaida);
	}

	/**
	 * Retorna o nome da função de atvação.
	 * @return nome da função de ativação.
	 */
	public String nome() {
		return getClass().getSimpleName();
	}
}
