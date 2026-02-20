package jnn.treino;

import java.util.LinkedList;
import java.util.concurrent.ForkJoinPool;

import jnn.core.parallel.PoolFactory;
import jnn.core.tensor.Tensor;
import jnn.dataloader.DataLoader;
import jnn.modelos.Modelo;
import jnn.treino.callback.CallbackFimEpoca;

/**
 * Interface para treino de modelos da biblioteca.
 */
public class Treinador implements Cloneable {

	/**
	 * Modelo base para treino.
	 */
	protected Modelo modelo;

	/**
	 * Formato de execução de treino (amostra-amostra, lotes).
	 */
	MetodoTreino metodo;

	/**
	 * Histórico de perda do modelo durante o treinamento.
	 */
	protected LinkedList<Float> historico = new LinkedList<>();

	/**
	 * Armazenar histórico de perda durante o treino.
	 */
	protected boolean calcHist;

	/**
	 * Seed para o gerador de número aleatórios.
	 */
	protected long seed = 0L;// seed padrão

	/**
	 * Callback de fim de época.
	 */
	private CallbackFimEpoca callback;

	/**
	 * Inicializa um novo treinador.
	 * @param modelo modelo base
	 * @param hist calcular histórico de perda.
	 */
	public Treinador(Modelo modelo, boolean hist) {
		this.modelo = modelo;
		this.calcHist = hist;
	}

	/**
	 * Iniciliza um novo treinador.
	 * @param modelo modelo base.
	 */
	public Treinador(Modelo modelo) {
		this(modelo, false);
	}

	/**
	 * Configura um novo método de treino.
	 * @param metodo novo método de treino.
	 */
	public void setMetodo(MetodoTreino metodo) {
		if (metodo != null) {
			this.metodo = metodo;
			this.metodo.historico = historico;
		}
	}

	/**
	 * Configura um callback para ser chamado a cada final de época.
	 * @param callback novo callback.
	 */
	public void setCallback(CallbackFimEpoca callback) {
		this.callback = callback;
	}

	/**
	 * Configura a seed do gerador de números aleatórios.
	 * @param seed nova seed.
	 */
	public void setSeed(Number seed) {
		this.seed = seed.longValue();
	}

	/**
	 * Configura o cálculo para o histórico de perdas durante o treinamento.
	 * @param calcular calcular ou não o histórico de custo.
	 */
	public void setHistorico(boolean calcular) {
		this.calcHist = calcular;
	}

	/**
	 * Executa a regra de treino durante um determinado número de épocas.
	 * @param xs {@code Tensores} contendos os dados de entrada.
	 * @param ys {@code Tensores} contendos os dados de saída (rótulos).
	 * @param epochs quantidade de épocas de treinamento.
	 * @param tamLote tamanho do lote de amostras.
	 * @param logs logs para perda durante as épocas de treinamento.
	 */
	public void executar(Tensor[] xs, Tensor[] ys, int epochs, int tamLote, boolean logs) {
		if (tamLote < 1) {
			throw new IllegalArgumentException(
				"\nTamanho de lote " + tamLote + " inválido."
			);
		}

		if (tamLote < 2) setMetodo(new Treino(modelo));
		else setMetodo(new TreinoLote(modelo, tamLote));

		metodo.setCallback(callback);

		metodo.calcHist = calcHist;
		
		// só mexer na seed se for mudado o valor padrão.
		if (seed != 0) metodo.random.setSeed(seed);
		
		modelo.treino(true);	
		metodo.loop(
			xs,
			ys,
			modelo.otm(),
			modelo.loss(),
			xs.length,
			epochs,
			logs
		);
		modelo.treino(false);
	}

	/**
	 * Executa a regra de treino durante um determinado número de épocas.
	 * @param loader {@code DataLoader} com conjunto de amostras.
	 * @param epochs quantidade de épocas de treinamento.
	 * @param logs logs para perda durante as épocas de treinamento.
	 * @see jnn.dataloader.DataLoader
	 */
	public void executar(DataLoader loader, int epochs, boolean logs) {
		executar(loader, epochs, 1, logs);
	}

	/**
	 * Executa a regra de treino durante um determinado número de épocas.
	 * @param loader {@code DataLoader} com conjunto de amostras.
	 * @param epochs quantidade de épocas de treinamento.
	 * @param tamLote tamanho do lote de amostras.
	 * @param logs logs para perda durante as épocas de treinamento.
	 * @see jnn.dataloader.DataLoader
	 */
	public void executar(DataLoader loader, int epochs, int tamLote, boolean logs) {
		executar(loader.getX(), loader.getY(), epochs, tamLote, logs);
	}

	/**
	 * Executa a regra de treino durante um determinado número de épocas.
	 * @param xs {@code Tensores} contendos os dados de entrada.
	 * @param ys {@code Tensores} contendos os dados de saída (rótulos).
	 * @param epochs quantidade de épocas de treinamento.
	 */
	public void executar(Tensor[] xs, Tensor[] ys, int epochs) {
		executar(xs, ys, epochs, 1, false);
	}

	/**
	 * Executa a regra de treino durante um determinado número de épocas.
	 * @param xs {@code Tensores} contendos os dados de entrada.
	 * @param ys {@code Tensores} contendos os dados de saída (rótulos).
	 * @param tamLote tamanho do lote de amostras.
	 * @param epochs quantidade de épocas de treinamento.
	 */
	public void executar(Tensor[] xs, Tensor[] ys, int tamLote, int epochs) {
		executar(xs, ys, epochs, tamLote, false);
	}

	/**
	 * Executa a regra de treino durante um determinado número de épocas.
	 * @param loader conjunto de dados.
	 * @param epochs quantidade de épocas de treinamento.
	 */
	public void executar(DataLoader loader, int epochs) {
		executar(loader, epochs, 1, false);
	}

	/**
	 * Realiza a retropropagação de gradientes através das camadas do modelo.
	 * @param g {@code Tensor} contendo o gradiente em relação a saída prevista
	 * pelo modelo.
	 */
	public void backpropagation(Tensor g) {
		metodo.backpropagation(g);
	}

	/**
	 * Retorna um array contendo os valores de perda por época de treinamento.
	 * @return lista de perdas do modelo.
	 */
	public float[] hist() {
		Float[] hist = metodo.hist();
		float[] h = new float[hist.length];

		final int t = Runtime.getRuntime().availableProcessors();
		try (ForkJoinPool pool = PoolFactory.pool(t)) {
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
	public String nome() {
		return getClass().getSimpleName();
	}
	
}
