package jnn.train.method;

import jnn.core.tensor.Tensor;
import jnn.dataloader.Sample;
import jnn.metrics.loss.Loss;
import jnn.dataloader.DataLoader;
import jnn.models.Model;
import jnn.otm.Optimizer;
import jnn.train.callback.EpochInfo;

/**
  * Implementação de treino dos modelos.
 */
public class RegularTrain extends TrainMethod {

	/**
	 * Treinador sequencial, atualiza os parâmetros a cada amostra de dados.
	 * @param model modelo para treino.
	 * @param hist calcular histórico de perda.
	 */
	public RegularTrain(Model model, boolean hist) {
		super(model, hist);
	}

	/**
	 * Treinador sequencial, atualiza os parâmetros a cada amostra de dados.
	 * @param model modelo para treino.
	 */
	public RegularTrain(Model model) {
		this(model, false);
	}

	@Override
	public void loop(DataLoader loader, Optimizer otm, Loss loss, int epochs, boolean logs) {
		if (logs) hideCursor();
		long tempo = 0;
		int largEpocas = String.valueOf(epochs).length();

		final int n = loader.len();
		for (int e = 0; e < epochs; e++) {
			if (logs) tempo = System.nanoTime();

			float perdaEpoca = 0.0f;
			loader.suffle();
			
			for (int i = 0; i < n; i++) {
				Sample sample = loader.get(i);
				Tensor prev = model.forward(sample.x());
				
				if (calculate) perdaEpoca += loss.forward(prev, sample.y()).item();
				
				model.zeroGrad();
				model.backward(loss.backward(prev, sample.y()));
				otm.update();
			}
			
			if (logs) {
				tempo = System.nanoTime() - tempo;
				clearLine();
				String perda = String.format("%.8f", (perdaEpoca/n)).replaceAll(",", ".");
				String log = String.format("(Época %" + largEpocas + "d/%" + largEpocas + "d) loss: %s", (e + 1), epochs, perda);

				long segundos = (long) tempo / 1_000_000_000;
				if (segundos < 60) {
					log += String.format(" (%ds)", segundos);
				
				} else {
					long min = (segundos / 60);
					long seg = segundos % 60;
					log += String.format(" (%dmin %ds)", min, seg);
				}

				showTrainLog(log);
			}

			if (calculate) history.add(perdaEpoca/n);

			if (scheduler != null) scheduler.update();
			if (callback != null)  callback.onEpochEnd(new EpochInfo(e, perdaEpoca));
		}

		if (logs) {
			showCursor();
			System.out.println();
		}

	}

}
