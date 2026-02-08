package jnn.treino;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;
import jnn.metrica.perda.Perda;
import jnn.modelos.Modelo;
import jnn.otm.Otimizador;
import jnn.treino.callback.InfoEpoca;

/**
  * Implementação de treino em lote dos modelos.
 */
public class TreinoLote extends MetodoTreino {

	/**
	 * Tamanho do lote de amostras por iteração
	 */
	int tamLote;

	/**
	 * Treinador em lotes, atualiza os parâmetros a subamostra
	 * do dataset fornecido.
	 * @param modelo modelo base.
	 * @param hist calcular histórico de perda.
	 * @param tamLote tamanho do lote de treino.
	 */
	public TreinoLote(Modelo modelo, boolean hist, int tamLote) {
		super(modelo, hist);
		this.tamLote = tamLote;
	}

	/**
	 * Treinador em lotes, atualiza os parâmetros a subamostra
	 * do dataset fornecido.
	 * @param modelo modelo base.
	 * @param tamLote tamanho do lote de treino.
	 */
	public TreinoLote(Modelo modelo, int tamLote) {
		this(modelo, false, tamLote);
	}

	@Override
	protected void loop(Tensor[] x, Tensor[] y, Otimizador otm, Perda loss, int amostras, int epochs, boolean logs) {
		if (logs) esconderCursor();
		long tempo = 0;

		for (int e = 0; e < epochs; e++) {
			if (logs) tempo = System.nanoTime();

			embaralhar(x, y);
			float perdaEpoca = 0.0f;

			for (int i = 0; i < amostras; i += tamLote) {
				int idFim = Math.min(i + tamLote, amostras);
				Tensor[] loteX = JNNutils.subArray(x, i, idFim);
				Tensor[] loteY = JNNutils.subArray(y, i, idFim);

                modelo.gradZero();
				perdaEpoca += processoLote(loteX, loteY, loss);
                otm.update();      
			}
			
			if (logs) {
				tempo = System.nanoTime() - tempo;

				limparLinha();
				String log = "[Época " + (e+1) + "/" + epochs + "] loss: " + (perdaEpoca/amostras);

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

			if (calcHist) historico.add(perdaEpoca / amostras);
			
			if (callback != null) {
				callback.run(new InfoEpoca(e, perdaEpoca));
			}
		}

		if (logs) {
			exibirCursor();
			System.out.println();
		}
	}

	/**
	 * Executa o passo de treino em lotes.
	 * @param loteX array de {@code Tensor} contendos entradas de treino.
	 * @param loteY array de {@code Tensor} contendos rótulos de treino.
	 * @param loss função de perda do modelo.
	 * @param perdaEpoca valor de perda por época de treinamento.
	 */
	private float processoLote(Tensor[] loteX, Tensor[] loteY, Perda loss) {
		Tensor real = JNNutils.concatenar(loteY);

		Tensor prev = modelo.forward(JNNutils.concatenar(loteX));
		Tensor g = loss.backward(prev, real);
		
		modelo.backward(g);

		if (calcHist) {
			float l = loss.forward(prev, real).item();
			int n = loteX.length;
			return l * n;
		}

		return 0.0f;// não registrar perda
	}

}
