package jnn.treinamento;

import java.util.LinkedList;
import java.util.Random;

import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;

/**
 * Interface para treino de modelos da biblioteca.
 */
public abstract class Treinador implements Cloneable {

	/**
	 * Modelo de treino.
	 */
	Modelo modelo;

	/**
	 * Gerador de números pseudo-aleatórios.
	 */
	Random random;

	/**
	 * Histórico de perda do modelo durante o treinamento.
	 */
	protected LinkedList<Double> historico;
	
	/**
	 * Variável de controle para armazenagem do histórico de treino.
	 */
	protected boolean calcularHistorico;

	/**
	 * Tamanho do lote de treinamento.
	 */
	protected int tamLote;

	/**
	 * Construtor implícito.
	 */
	protected Treinador(Modelo modelo, int tamLote) {
		this.modelo = modelo;

		random = new Random();
		historico = new LinkedList<>();
		calcularHistorico = false;
		this.tamLote = tamLote;
	}

	/**
	 * Construtor implícito.
	 */
	protected Treinador(Modelo modelo) {
		this(modelo, 1);
	}

	/**
	 * Configura a seed do gerador de números aleatórios.
	 * @param seed nova seed.
	 */
	public void setSeed(Number seed) {
		if (seed != null) {
			random.setSeed(seed.longValue());
		}
	}

	/**
	 * Configura o cálculo para o histórico de perdas durante o treinamento.
	 * @param calcular calcular ou não o histórico de custo.
	 */
	public void setHistorico(boolean calcular) {
		calcularHistorico = calcular;
	}

	/**
	 * Executa a regra de treino durante um determinado número de épocas.
	 * @param x {@code Tensores} contendos os dados de entrada.
	 * @param y {@code Tensores} contendos os dados de saída (rótulos).
	 * @param epochs quantidade de épocas de treinamento.
	 * @param logs logs para perda durante as épocas de treinamento.
	 */
	public abstract void executar(Tensor[] x, Tensor[] y, int epochs, boolean logs);

	/**
	 * Executa a regra de treino durante um determinado número de épocas.
	 * @param x {@code Tensores} contendos os dados de entrada.
	 * @param y {@code Tensores} contendos os dados de saída (rótulos).
	 * @param epochs quantidade de épocas de treinamento.
	 */
	public void executar(Tensor[] x, Tensor[] y, int epochs) {
		executar(x, y, epochs, false);
	}

	/**
	 * Realiza a retropropagação de gradientes de cada camada para a atualização de seus parâmetros.
	 * <p>
	 *    Os gradientes iniciais são calculados usando a derivada da função de perda em relação
	 *    aos erros do modelo.
	 * </p>
	 * <p>
	 *    A partir disso, são retropropagados de volta da última camada do modelo até a primeira.
	 * </p>
	 * @param prev {@code Tensor} contendos os dados previstos.
	 * @param real {@code Tensor} contendos os dados reais (rotulados).
	 */
	public void backpropagation(Tensor prev, Tensor real) {
		Tensor grad = modelo.perda().derivada(prev, real);
		
		final int n = modelo.numCamadas();
		for (int i = n-1; i >= 0; i--) {
			grad = modelo.camada(i).backward(grad);
		}
	}

	/**
	 * Embaralha os dos arrays usando o algoritmo Fisher-Yates.
	 * @param <T> tipo de dados de entrada e saida.
	 * @param x {@code array} com os dados de entrada.
	 * @param y {@code array} com os dados de saída.
	 */
	public <T> void embaralhar(T[] x, T[] y) {
		int linhas = x.length;
		int i, idAleatorio;

		T temp;
		for (i = linhas - 1; i > 0; i--) {
			idAleatorio = random.nextInt(i+1);
			
			// entradas
			temp = x[i];
			x[i] = x[idAleatorio];
			x[idAleatorio] = temp;

			// saídas
			temp = y[i];
			y[i] = y[idAleatorio];
			y[idAleatorio] = temp;
		}
	}

	/**
	 * Retorna um array contendo os valores de perda por época de treinamento.
	 * @return lista de perdas do modelo.
	 */
	public double[] historico() {
		Object[] hist = historico.toArray();
		double[] h = new double[hist.length];

		for (int i = 0; i < h.length; i++) {
			h[i] = (double) hist[i];
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

	@Override
	public Treinador clone() {
		try {
			return (Treinador) super.clone();
		} catch (CloneNotSupportedException e) {
			throw new RuntimeException(e);
		}
	}

	/** 
	 * Esconde o cursor do terminal.
	 */
	public void esconderCursor() {
		System.out.print("\033[?25l");
	}

	/**
	 * Exibe o cursor no terminal.
	 */
	public void exibirCursor() {
		System.out.print("\033[?25h");
	}

	/**
	 * Atualiza as informações do log de treino.
	 * @param log informações desejadas.
	 */
	public void exibirLogTreino(String log) {
		System.out.println(log);
		System.out.print("\033[1A"); // mover pra a linha anterior
	}

	/**
	 * Limpa a linha de log
	 */
	public void limparLinha() {
		System.out.print("\033[2K");
	}
	
}
