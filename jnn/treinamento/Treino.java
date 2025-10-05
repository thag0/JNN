package jnn.treinamento;

import jnn.avaliacao.perda.Perda;
import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;
import jnn.otimizadores.Otimizador;

/**
  * Implementação de treino dos modelos.
 */
public class Treino extends Treinador {

	/**
	 * Treinador sequencial, atualiza os parâmetros a cada amostra de dados.
	 * @param modelo modelo para treino.
	 */
	public Treino(Modelo modelo) {
		super(modelo);
	}

	@Override
	protected void loop(Tensor[] x, Tensor[] y, Otimizador otm, Perda loss, int amostras, int epochs, boolean logs) {
		modelo.treino(true);

		if (logs) esconderCursor();
		double perdaEpoca;
		for (int e = 1; e <= epochs; e++) {
			embaralhar(x, y);
			perdaEpoca = 0;
			
			for (int i = 0; i < amostras; i++) {
				Tensor prev = modelo.forward(x[i]);
				
				//feedback de avanço
				if (calcularHistorico) {
					perdaEpoca += loss.forward(prev, y[i]).item();
				}
				
				modelo.gradZero();
				backpropagation(loss.backward(prev, y[i]));
				otm.atualizar();
			}

			if (logs) {
				limparLinha();
				exibirLogTreino("Época " +  e + "/" + epochs + " -> perda: " + (double)(perdaEpoca/amostras));
			}

			// feedback de avanço
			if (calcularHistorico) historico.add(perdaEpoca/amostras);
		}

		if (logs) {
			exibirCursor();
			System.out.println();
		}

		modelo.treino(false);
	}

}
