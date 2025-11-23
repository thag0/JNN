package jnn.acts;

import jnn.camadas.Conv2D;
import jnn.camadas.Densa;
import jnn.core.tensor.Tensor;

/**
 * Implementação da função de ativação Softmax para uso
 * dentro dos modelos.
 */
public class Softmax extends Ativacao {

	/**
	 * Instancia a função de ativação Softmax.
	 * <p>
	 *    A função Softmax transforma os valores de entrada em probabilidades
	 *    normalizadas,
	 *    permitindo que a unidade com a maior saída tenha uma probabilidade mais
	 *    alta.
	 * </p>
	 * Exemplo:
	 * <pre>
	 * tensor = [
	 *    [1, 2, 3]
	 *]
	 *
	 *softmax.forward(tensor, tensor);
	 *
	 *tensor = [
	 *    [0.090031, 0.244728, 0.665241]
	 *]
	 * </pre>
	 * <p>
	 *		Para tensores com duas dimensões, a softmax é aplicada a cada
	 *		"linha" do tensor.
	 * </p>
	 * <p>
	 * 		Para tensores com mais de três dimensões (3D+), a função não
	 * 		possui suporte.
	 * </p>
	 */
	public Softmax() {}

	@Override
	public void forward(Tensor x, Tensor dest) {
		if (x.compShape(dest) == false) {
			throw new IllegalArgumentException(
				"\nFormato do tensor de entrada (" + x.shapeStr() + ") " +
				"deve ser igual ao formato do tensor de saída (" + dest.shapeStr() + ")"
			);				
		}

		int[] shape = x.shape();
		if (shape.length == 1) {
			forward1D(x, dest);

		} else if (shape.length == 2) {
			int lin = shape[0];
			for (int i = 0; i < lin; i++) {
				forward1D(x.subTensor(i), dest.subTensor(i));
			}

		} else {
			throw new UnsupportedOperationException(
				"\nSuporte apenas para tensores 1D e 2D."
			);
		}
	}

	/**
	 * Função interna pra calcular softmax
	 * @param x {@code Tensor} de entrada.
	 * @param dst {@code Tensor} de destino.
	 */
	private void forward1D(Tensor x, Tensor dst) {
		int tam = x.tamDim(0);
		double soma = 0;
		
		for (int i = 0; i < tam; i++) {
			soma += Math.exp(x.get(i));
		}

		for (int i = 0; i < tam; i++) {
			double s = Math.exp(x.get(i)) / soma;
			dst.set(s, i);
		}	
	}

	@Override
	public void backward(Densa camada) {
		final int numDim = camada._saida.numDim();
		
		if (numDim == 1) {
			backward1D(camada._saida, camada._gradSaida);
			
		} else if (numDim == 2) {
			final int lin = camada._saida.tamDim(0);
			for (int i = 0; i < lin; i++) {
				backward1D(
					camada._saida.subTensor(i),
					camada._gradSaida.subTensor(i)
				);
			}
		
		} else {
			throw new UnsupportedOperationException(
				"\nSem suporte."
			);
		}

	}

	/**
	 * Função interna pra retropropagar softmax.
	 * @param softmax saida da camada (resultado do softmax).
	 * @param g gradiente de saída da camada.
	 */
	private void backward1D(Tensor softmax, Tensor g) {
		int tam = softmax.tamDim(0);

		double p = 0;
		for (int i = 0; i < tam; i++) {
			p += g.get(i) * softmax.get(i);
		}

		for (int i = 0; i < tam; i++) {
			double gi = softmax.get(i) * (g.get(i) - p);
			g.set(gi, i);
		}
	}

	@Override
	public void backward(Conv2D camada) {
		throw new UnsupportedOperationException(
			"\nSem suporte para Conv2D."
		);
	}

}
