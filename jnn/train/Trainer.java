package jnn.train;

import java.util.LinkedList;
import java.util.concurrent.ForkJoinPool;

import jnn.core.JNNutils;
import jnn.core.parallel.JNNparallel;
import jnn.dataloader.DataLoader;
import jnn.models.Model;
import jnn.otm.scheduler.Scheduler;
import jnn.train.callback.Callback;
import jnn.train.method.BatchTrain;
import jnn.train.method.RegularTrain;
import jnn.train.method.TrainMethod;

/**
 * Interface para treino de modelos da biblioteca.
 */
public class Trainer implements Cloneable {

	/**
	 * Modelo base para treino.
	 */
	protected Model model;

	/**
	 * Formato de execução de treino (amostra-amostra, lotes).
	 */
	TrainMethod method;

	/**
	 * Histórico de perda do modelo durante o treinamento.
	 */
	protected LinkedList<Float> history = new LinkedList<>();

	/**
	 * Armazenar histórico de perda durante o treino.
	 */
	protected boolean calculateHistory;

	/**
	 * Callback de fim de época.
	 */
	private Callback callback;

	/**
	 * Agendador para lr dinâmico dos otimizadores.
	 */
	private Scheduler scheduler;

	/**
	 * Inicializa um novo treinador.
	 * @param model modelo base
	 * @param history calcular histórico de perda.
	 */
	public Trainer(Model model, boolean history) {
		this.model = JNNutils.validarNaoNulo(model, "modelo == null");
		this.calculateHistory = history;
	}

	/**
	 * Iniciliza um novo treinador.
	 * @param model modelo base.
	 */
	public Trainer(Model model) {
		this(model, false);
	}

	/**
	 * Configura um novo método de treino.
	 * @param method novo método de treino.
	 */
	public void setMethod(TrainMethod method) {
		if (method != null) {
			this.method = method;
			this.method.history = history;
		}
	}

	/**
	 * Configura um callback para ser chamado a cada final de época.
	 * @param callback novo callback.
	 */
	public void setCallback(Callback callback) {
		this.callback = callback;
	}

	/**
	 * Configura um scheduler para ser chamado a cada final de época.
	 * @param scheduler novo scheduler.
	 */
	public void setScheduler(Scheduler scheduler) {
		this.scheduler = scheduler;
	}

	/**
	 * Configura o cálculo para o histórico de perdas durante o treinamento.
	 * @param calculate calcular ou não o histórico de custo.
	 */
	public void setHistory(boolean calculate) {
		this.calculateHistory = calculate;
	}

	/**
	 * Executa a regra de treino durante um determinado número de épocas.
	 * @param loader {@code DataLoader} com conjunto de dados.
	 * @param epochs quantidade de épocas de treinamento.
	 * @param batchSize tamanho do lote de amostras.
	 * @param logs logs para perda durante as épocas de treinamento.
	 */
	public void execute(DataLoader loader, int epochs, int batchSize, boolean logs) {
		if (batchSize < 1) {
			throw new IllegalArgumentException(
				"\nTamanho de lote " + batchSize + " inválido."
			);
		}

		if (batchSize < 2) setMethod(new RegularTrain(model));
		else setMethod(new BatchTrain(model, batchSize));

		method.setCallback(callback);
		method.setScheduler(scheduler);

		method.calculate = calculateHistory;
		
		method.loop(
			loader,
			model.optim(),
			model.loss(),
			epochs,
			logs
		);
	}

	/**
	 * Executa a regra de treino durante um determinado número de épocas.
	 * @param loader {@code DataLoader} com conjunto de dados.
	 * @param epochs quantidade de épocas de treinamento.
	 * @param logs logs para perda durante as épocas de treinamento.
	 * @see jnn.dataloader.DataLoader
	 */
	public void execute(DataLoader loader, int epochs, boolean logs) {
		execute(loader, epochs, 1, logs);
	}

	/**
	 * Executa a regra de treino durante um determinado número de épocas.
	 * @param loader conjunto de dados.
	 * @param epochs quantidade de épocas de treinamento.
	 */
	public void execute(DataLoader loader, int epochs) {
		execute(loader, epochs, 1, false);
	}

	/**
	 * Retorna um array contendo os valores de perda por época de treinamento.
	 * @return lista de perdas do modelo.
	 */
	public float[] hist() {
		Float[] hist = method.hist();
		float[] h = new float[hist.length];

		final int t = Runtime.getRuntime().availableProcessors();
		try (ForkJoinPool pool = JNNparallel.pool(t)) {
			for (int i = 0; i < h.length; i++) {
				final int id = i;
				pool.execute(() -> h[id] = hist[id]);
			}
		}

		return h;
	}
	
	/**
	 * Retorna o nome do treinador.
	 * @return nome do treinador.
	 */
	public String name() {
		return getClass().getSimpleName();
	}
	
}
