package jnn.ativacoes;

import java.util.function.DoubleUnaryOperator;

import jnn.camadas.Conv2D;
import jnn.camadas.Densa;
import jnn.core.Utils;
import jnn.core.tensor.Tensor;

/**
 * Classe base para a implementação das funções de ativação.
 * <p>
 *    As funções de ativação percorres todos os elementos contendo os
 *    resultados de cada operação dos kernels das camadas, e aplica sua
 *    operação correspondente nas suas saídas.
 * </p>
 * <p>
 *    Funções de ativação podem fazer uso dos métodos {@code aplicarFx()} e 
 *    {@code aplicarDx()}, sendo necessário informar nos seus constritures 
 *    uma interface funcional que fará o cálculo da saída de acordo com uma
 *    entrada redebida.
 * </p>
 * Exemplo com a função ReLU:
 * <pre>
 *public class ReLU extends Ativacao{
 *  public ReLU(){
 *    super.construir(
 *       (x) -> (x > 0) ? x : 0,
 *       (x) -> (x > 0) ? 1 : 0
 *    );
 *  }
 *
 *}
 * </pre>
 * <p>
 *    Novas funções de ativações devem sobrescrever os métodos existentes {@code ativar()} e {@code derivada()}.
 * </p>
 */
public abstract class Ativacao {

	/**
	 * Função de ativação.
	 */
	protected DoubleUnaryOperator fx;

	/**
	 * Derivada da função de ativação.
	 */
	protected DoubleUnaryOperator dx;

	/**
	 * Utilitário.
	 */
	protected Utils utils = new Utils();

	/**
	 * Configura a função de ativação e sua derivada para uso.
	 * @param fx função de ativação.
	 * @param dx deriviada da função de ativação
	 */
	public void construir(DoubleUnaryOperator fx, DoubleUnaryOperator dx) {
		utils.validarNaoNulo(fx, "Função de ativação não pode ser nula.");

		this.fx = fx;
		this.dx = dx;
	}

	/**
	 * Calcula o resultado da ativação de acordo com a função configurada.
	 * @param entrada {@code Tensor} de entrada.
	 * @param dest {@code Tensor} de destino.
	 */
	public void forward(Tensor entrada, Tensor dest) {
		dest.aplicar(entrada, fx);
	}

	/**
	 * Calcula o resultado da derivada da função de ativação de acordo 
	 * com a função configurada
	 * @param entrada {@code Tensor} de entrada.
	 * @param grad {@code Tensor} contendo os gradientes.
	 * @param dest {@code Tensor} de destino.
	 */
	public void backward(Tensor entrada, Tensor grad, Tensor dest) {
		dest.aplicar(entrada, grad, (e, g) -> dx.applyAsDouble(e) * g);
	}

	/**
	 * Implementação especifíca para camadas densas.
	 * <p>
	 *    Função criada para dar suporte a ativações especiais.
	 * </p>
	 * @param camada camada densa.
	 */
	public void backward(Densa camada) {
		//por padrão chamar o método da própria ativação
		backward(camada._somatorio, camada._gradSaida, camada._gradSaida);
	}

	/**
	 * Implementação especifíca para camadas convolucionais.
	 * <p>
	 *    Função criada para dar suporte a ativações especiais.
	 * </p>
	 * @param camada camada convolucional.
	 */
	public void backward(Conv2D camada) {
		//por padrão chamar o método da própria ativação
		backward(camada._somatorio, camada._gradSaida, camada._gradSaida);
	}

	/**
	 * Retorna o nome da função de atvação.
	 * @return nome da função de ativação.
	 */
	public String nome() {
		return getClass().getSimpleName();
	}
}
