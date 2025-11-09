package jnn.treinamento;

import jnn.avaliacao.perda.Perda;
import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;
import jnn.otimizadores.Otimizador;

/**
  * Implementação de treino dos modelos.
 */
public class Treino extends MetodoTreino {

	/**
	 * Treinador sequencial, atualiza os parâmetros a cada amostra de dados.
	 * @param modelo modelo para treino.
	 */
	public Treino(Modelo modelo, boolean hist) {
		super(modelo, hist);
	}

	public Treino(Modelo modelo) {
		this(modelo, false);
	}

	@Override
	protected void loop(Tensor[] x, Tensor[] y, Otimizador otm, Perda loss, int amostras, int epochs, boolean logs) {
		if (logs) esconderCursor();
		for (int e = 1; e <= epochs; e++) {
			double perdaEpoca = 0;
			embaralhar(x, y);
			
			for (int i = 0; i < amostras; i++) {
				Tensor prev = modelo.forward(x[i]);
				
				//feedback de avanço
				if (calcHist) perdaEpoca += loss.forward(prev, y[i]).item();
				
				modelo.gradZero();
				backpropagation(loss.backward(prev, y[i]));
				otm.update();
			}

			if (logs) {
				limparLinha();
				exibirLogTreino("Época " +  e + "/" + epochs + " -> perda: " + (double)(perdaEpoca/amostras));
			}

			// feedback de avanço
			if (calcHist) historico.add(perdaEpoca/amostras);
		}

		if (logs) {
			exibirCursor();
			System.out.println();
		}

	}

}
