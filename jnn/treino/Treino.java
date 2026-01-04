package jnn.treino;

import jnn.core.tensor.Tensor;
import jnn.metrica.perda.Perda;
import jnn.modelos.Modelo;
import jnn.otm.Otimizador;

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
		long tempo = 0;

		for (int e = 1; e <= epochs; e++) {
			if (logs) tempo = System.nanoTime();

			double perdaEpoca = 0;
			embaralhar(x, y);
			
			for (int i = 0; i < amostras; i++) {
				Tensor prev = modelo.forward(x[i]);
				
				if (calcHist) perdaEpoca += loss.forward(prev, y[i]).item();
				
				modelo.gradZero();
				backpropagation(loss.backward(prev, y[i]));
				otm.update();
			}
			
			if (logs) {
				tempo = System.nanoTime() - tempo;
				limparLinha();
				String log = "Época " +  e + "/" + epochs + " -> perda: " + (float)(perdaEpoca/amostras);

				long segundos = (long) tempo / 1_000_000_000;
				long min = (segundos / 60);
				long seg = segundos % 60;
				if (segundos < 60) {
					log += String.format(" (%ds)", segundos);
				} else {
					log += String.format(" (%dmin %ds)", min, seg);
				}

				exibirLogTreino(log);
			}

			if (calcHist) historico.add(perdaEpoca/amostras);
		}

		if (logs) {
			exibirCursor();
			System.out.println();
		}

	}

}
