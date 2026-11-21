package jnn.train.method;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;
import jnn.dataloader.DataLoader;
import jnn.metrics.loss.Loss;
import jnn.dataloader.Batch;
import jnn.models.Model;
import jnn.otm.Optimizer;
import jnn.train.callback.EpochInfo;

/**
  * Implementação de treino em lote dos modelos.
 */
public class BatchTrain extends TrainMethod {

	/**
	 * Tamanho do lote de amostras por iteração
	 */
	int batchSize;

	/**
	 * Treinador em lotes, atualiza os parâmetros a subamostra
	 * do dataset fornecido.
	 * @param model modelo base.
	 * @param hist calcular histórico de perda.
	 * @param batchSize tamanho do lote de treino.
	 */
	public BatchTrain(Model model, boolean hist, int batchSize) {
		super(model, hist);
		this.batchSize = batchSize;
	}

	/**
	 * Treinador em lotes, atualiza os parâmetros a subamostra
	 * do dataset fornecido.
	 * @param model modelo base.
	 * @param batchSize tamanho do lote de treino.
	 */
	public BatchTrain(Model model, int batchSize) {
		this(model, false, batchSize);
	}

	@Override
	public void loop(DataLoader loader, Optimizer otm, Loss loss, int epochs, boolean logs) {
		if (logs) hideCursor();
		long tempo = 0;
		int largEpocas = String.valueOf(epochs).length();

		final int n = loader.len();
		for (int e = 0; e < epochs; e++) {
			if (logs) tempo = System.nanoTime();

			loader.suffle();
			float perdaEpoca = 0.0f;

			for (int i = 0; i < n; i += batchSize) {
				Batch lote = loader.batch(i, batchSize);

                model.zeroGrad();
				perdaEpoca += processoLote(lote.arrX(), lote.arrY(), loss);
                otm.update();      
			}
			
			if (logs) {
				tempo = System.nanoTime() - tempo;

				clearLine();
				String perda = String.format("%.8f", (perdaEpoca/n)).replaceAll(",", ".");
				String log = String.format("(Época %" + largEpocas + "d/%" + largEpocas + "d) loss: %s", (e + 1), epochs, perda);

				long segundos = (long) tempo / 1_000_000_000;
				long min = (segundos / 60);
				long seg = segundos % 60;
				if (segundos < 60) {
					log += String.format(" (%ds)", segundos);
				} else {
					log += String.format(" (%dmin %ds)", min, seg);
				}

				showTrainLog(log);
			}

			if (calculate) history.add(perdaEpoca / n);

			if (scheduler != null) scheduler.update();
			if (callback != null)  callback.onEpochEnd(new EpochInfo(e, perdaEpoca));
		}

		if (logs) {
			showCursor();
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
	private float processoLote(Tensor[] loteX, Tensor[] loteY, Loss loss) {
		Tensor xs = JNNutils.concat(loteX);
		Tensor ys = JNNutils.concat(loteY);
		
		Tensor y = model.forward(xs);
		Tensor g = loss.backward(y, ys);
		
		model.backward(g);

		if (calculate) {
			float l = loss.forward(y, ys).item();
			int n = loteX.length;
			return l * n;
		}

		return 0.0f;// não registrar perda
	}

}
