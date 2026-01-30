package jnn.acts;

import jnn.camadas.Conv2D;
import jnn.camadas.Densa;
import jnn.core.tensor.Tensor;

/**
 * Classe base para a implementação das funções de ativação.
 * <p>
 *		As funções de ativação são usadas para melhorar a capacidade de modelagem
 *		dos dados em que os modelos treinados estão sendo usados.
 * </p>
 * <p>
 * 		Novas implementações devem obrigatoriamente informar o cálculo correspondente
 * 		da ativação, usando o método {@code construir()}, onde nele é informado tanto
 * 		a expressão da função de ativação, quanto sua derivada.
 * </p>
 * Exemplo:
 * <pre>
 *public class ReLU extends Ativacao{
 *  public ReLU(){
 *    construir(
 *       (x) -> (x > 0) ? x : 0,
 *       (x) -> (x > 0) ? 1 : 0
 *    );
 *  }
 *
 *}
 * </pre>
 * <p>
 *		Novas funções de ativações devem sobrescrever os métodos existentes 
 * 		{@code forward()} e {@code backward()}.
 * </p>
 */
public abstract class Ativacao {

	protected float fx(float x) {
		throw new UnsupportedOperationException("\nNão implementado.");
	}

	protected float dx(float x) {
		throw new UnsupportedOperationException("\nNão implementado.");
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
	 * @param camada camada densa.
	 */
	public void backward(Densa camada) {
		//por padrão chamar o método da própria ativação
		backward(camada._buffer, camada._gradSaida, camada._gradSaida);
	}

	/**
	 * Implementação especifíca para camadas convolucionais.
	 * <p>
	 *    Criada para dar suporte a ativações especiais.
	 * </p>
	 * @param camada camada convolucional.
	 */
	public void backward(Conv2D camada) {
		//por padrão chamar o método da própria ativação
		backward(camada._buffer, camada._gradSaida, camada._gradSaida);
	}

	/**
	 * Retorna o nome da função de atvação.
	 * @return nome da função de ativação.
	 */
	public String nome() {
		return getClass().getSimpleName();
	}
}
