package jnn.train.method;

import java.util.LinkedList;

import jnn.core.JNNlog;
import jnn.core.JNNutils;
import jnn.dataloader.DataLoader;
import jnn.metrics.loss.Loss;
import jnn.models.Model;
import jnn.otm.Otimizador;
import jnn.otm.scheduler.Scheduler;
import jnn.train.callback.Callback;

/**
 * Método base de treino de modelo.
 */
public abstract class TrainMethod {

	/**
	 * Modelo base para treino.
	 */
	protected Model model;

	/**
	 * Histórico de perda do modelo durante o treinamento.
	 */
	public LinkedList<Float> history;

	/**
	 * Variável de controle para armazenagem do histórico de treino.
	 */
	public boolean calculate;
	
	/**
	 * Callback usada a cada final de época.
	 */
	Callback callback;

	/**
	 * Scheduler usado a cada final de época.
	 */
	Scheduler scheduler;

	/**
	 * Construtor interno.
	 * @param model modelo base.
	 * @param calcHistory calcular histórico de perda durante o treino.
	 */
	protected TrainMethod(Model model, boolean calcHistory) {
		this.model = model;// já é validado != null pelo Treinador
		this.calculate = calcHistory;
	}

	/**
	 * Construtor interno.
	 * @param model modelo base.
	 */
	protected TrainMethod(Model model) {
		this(model, false);	
	}

	/**
	 * Configura um callback para ser chamado a cada final de época.
	 * @param callback novo callback.
	 */
	public void setCallback(Callback callback) {
		if (callback != null) this.callback = callback;
	}

	/**
	 * Configura um scheduler para ser chamado a cada final de época.
	 * @param scheduler novo scheduler.
	 */
	public void setScheduler(Scheduler scheduler) {
		if (scheduler != null) this.scheduler = scheduler;
	}

	/**
	 * Loop principal de treino.
	 * @param loader {@code DataLoader} com conjunto de dados.
	 * @param otm otimizador.
	 * @param loss função de perda.
	 * @param epochs quantidade de épocas de treinamento.
	 * @param logs exibir logs de avanço;
	 */
	public abstract void loop(DataLoader loader, Otimizador otm, Loss loss, int epochs, boolean logs);

	/**
	 * Embaralha ambos os arrays de entrada e saída.
	 * @param <T> tipo de dados de entrada e saida.
	 * @param xs {@code array} com dados de entrada.
	 * @param ys {@code array} com dados de saída.
	 */
	protected <T> void shuffle(T[] xs, T[] ys) {
		JNNutils.shuffle(xs, ys, null);
	}

	/** 
	 * Esconde o cursor do terminal.
	 */
	protected void hideCursor() {
		System.out.print("\033[?25l");
	}

	/**
	 * Exibe o cursor no terminal.
	 */
	protected void showCursor() {
		System.out.print("\033[?25h");
	}

	/**
	 * Atualiza as informações do log de treino.
	 * @param log informações desejadas.
	 */
	protected void showTrainLog(String log) {
		clearLine();
		JNNlog.trainLog(log);
	}

	/**
	 * Limpa a linha de log
	 */
	protected void clearLine() {
		System.out.print("\033[2K");
	}

	/**
	 * Retorna os valores do histórico de perdas do modelo durante o treino.
	 * @return histórico convertido em array de {@code Float}.
	 */
	public Float[] hist() {
		return history.toArray(new Float[]{});
	}
	
}
