package jnn.treino;

import java.util.LinkedList;
import java.util.concurrent.ForkJoinPool;

import jnn.core.parallel.PoolFactory;
import jnn.core.tensor.Tensor;
import jnn.dataloader.DataLoader;
import jnn.modelos.Modelo;

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
	protected LinkedList<Double> historico = new LinkedList<>();

	/**
	 * Número de threads para execução em paralelo.
	 */
	protected int _numThreads;

	/**
	 * Armazenar histórico de perda durante o treino.
	 */
	protected boolean calcHist;

	/**
	 * Seed para o gerador de número aleatórios.
	 */
	protected long seed = 0L;// seed padrão

	/**
	 * 
	 */
	public Treinador(Modelo modelo, boolean hist, int numThreads) {
		this.modelo = modelo;
		this._numThreads = numThreads;
		this.calcHist = hist;
	}

	/**
	 * 
	 */
	public Treinador(Modelo modelo) {
		this(modelo, false, 1);
	}

	public void setMetodo(MetodoTreino metodo) {
		if (metodo != null) {
			this.metodo = metodo;
			this.metodo.historico = historico;
		}
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
	 * Ajusta a quantidade de threads para utilizar durante o treino.
	 * @param n threads desejadas.
	 */
	public void setThreads(int n) {
		if (n < 1) {
			throw new IllegalArgumentException(
				"\nNúmero de threads " + n + " inválido."
			);
		}

		_numThreads = n;
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

		metodo.calcHist = calcHist;
		metodo._threads = _numThreads;
		
		// só mexer na seed se for mudada
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
	 * @see {@link jnn.dataloader.DataLoader}
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
	 * @see {@link jnn.dataloader.DataLoader}
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
	 * @param xs {@code Tensores} contendos os dados de entrada.
	 * @param ys {@code Tensores} contendos os dados de saída (rótulos).
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
	public double[] hist() {
		Double[] hist = metodo.hist();
		double[] h = new double[hist.length];

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
