package jnn.treino;

import jnn.core.tensor.Tensor;
import jnn.dataloader.Amostra;
import jnn.dataloader.DataLoader;
import jnn.metrica.perda.Perda;
import jnn.modelos.Modelo;
import jnn.otm.Otimizador;
import jnn.treino.callback.InfoEpoca;

/**
  * Implementação de treino dos modelos.
 */
public class Treino extends MetodoTreino {

	/**
	 * Treinador sequencial, atualiza os parâmetros a cada amostra de dados.
	 * @param modelo modelo para treino.
	 * @param hist calcular histórico de perda.
	 */
	public Treino(Modelo modelo, boolean hist) {
		super(modelo, hist);
	}

	/**
	 * Treinador sequencial, atualiza os parâmetros a cada amostra de dados.
	 * @param modelo modelo para treino.
	 */
	public Treino(Modelo modelo) {
		this(modelo, false);
	}

	@Override
	protected void loop(DataLoader loader, Otimizador otm, Perda loss, int epochs, boolean logs) {
		if (logs) esconderCursor();
		long tempo = 0;

		final int n = loader.tam();
		for (int e = 0; e < epochs; e++) {
			if (logs) tempo = System.nanoTime();

			float perdaEpoca = 0.0f;
			loader.embaralhar();
			
			for (int i = 0; i < n; i++) {
				Amostra a = loader.get(i);
				Tensor prev = modelo.forward(a.x());
				
				if (calcHist) perdaEpoca += loss.forward(prev, a.y()).item();
				
				modelo.gradZero();
				backpropagation(loss.backward(prev, a.y()));
				otm.update();
			}
			
			if (logs) {
				tempo = System.nanoTime() - tempo;
				limparLinha();
				String log = "Época " + (e+1) + "/" + epochs + " -> perda: " + (perdaEpoca/n);

				long segundos = (long) tempo / 1_000_000_000;
				if (segundos < 60) {
					log += String.format(" (%ds)", segundos);
				
				} else {
					long min = (segundos / 60);
					long seg = segundos % 60;
					log += String.format(" (%dmin %ds)", min, seg);
				}

				exibirLogTreino(log);
			}

			if (calcHist) historico.add(perdaEpoca/n);

			if (callback != null) {
				callback.run(new InfoEpoca(e, perdaEpoca));
			}
		}

		if (logs) {
			exibirCursor();
			System.out.println();
		}

	}

}
