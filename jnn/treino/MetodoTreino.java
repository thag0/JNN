package jnn.treino;

import java.util.LinkedList;
import java.util.Random;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;
import jnn.metrica.perda.Perda;
import jnn.modelos.Modelo;
import jnn.otm.Otimizador;

public abstract class MetodoTreino {

	/**
	 * Modelo base para treino.
	 */
	protected Modelo modelo;

	/**
	 * Histórico de perda do modelo durante o treinamento.
	 */
	protected LinkedList<Double> historico;
    
	/**
	 * Gerador de números pseudo-aleatórios.
	 */
	protected Random random;

	/**
	 * Quantidade de threads usadas pelo treinador.
	 */
	protected int _threads = 1;

	/**
	 * Variável de controle para armazenagem do histórico de treino.
	 */
	protected boolean calcHist;

	/**
	 * Construtor interno.
	 * @param modelo modelo base.
	 * @param hist calcular histórico de perda durante o treino.
	 */
	protected MetodoTreino(Modelo modelo, boolean hist) {
		this.modelo = modelo;
		this.calcHist = hist;
	}

	/**
	 * Construtor interno.
	 * @param modelo modelo base.
	 */
	protected MetodoTreino(Modelo modelo) {
		this(modelo, false);	
	}

	/**
	 * Configura uma seed manual para o método de treino, útil para replicar
	 * e comparar resultados.
	 * @param seed nova seed.
	 */
	public void setSeed(Number seed) {
		if (seed != null) {
			random.setSeed(seed.longValue());
		}
	}

	/**
	 * Realiza a retropropagação de gradientes através das camadas do modelo.
	 * @param g {@code Tensor} contendo o gradiente em relação a saída prevista
	 * pelo modelo.
	 */
	protected void backpropagation(Tensor g) {
		final int n = modelo.numCamadas();
		for (int i = n-1; i >= 0; i--) {
			g = modelo.camada(i).backward(g);
		}
	}

	/**
	 * Loop principal de treino.
	 * @param x {@code array} de {@code Tensor} com dados de entrada.
	 * @param y {@code array} de {@code Tensor} com dados de saída.
	 * @param otm otimizador.
	 * @param loss função de perda.
	 * @param amostras quantidade de amostras.
	 * @param epochs quantidade de épocas de treinamento.
	 * @param logs exibir logs de avanço;
	 */
	protected abstract void loop(Tensor[] x, Tensor[] y, Otimizador otm, Perda loss, int amostras, int epochs, boolean logs);

	/**
	 * Embaralha ambos os arrays de entrada e saída.
	 * @param <T> tipo de dados de entrada e saida.
	 * @param xs {@code array} com dados de entrada.
	 * @param ys {@code array} com dados de saída.
	 */
	protected <T> void embaralhar(T[] xs, T[] ys) {
		JNNutils.embaralhar(xs, ys, random);
	}

	/** 
	 * Esconde o cursor do terminal.
	 */
	protected void esconderCursor() {
		System.out.print("\033[?25l");
	}

	/**
	 * Exibe o cursor no terminal.
	 */
	protected void exibirCursor() {
		System.out.print("\033[?25h");
	}

	/**
	 * Atualiza as informações do log de treino.
	 * @param log informações desejadas.
	 */
	protected void exibirLogTreino(String log) {
		System.out.println(log);
		System.out.print("\033[1A"); // mover pra a linha anterior
	}

	/**
	 * Limpa a linha de log
	 */
	protected void limparLinha() {
		System.out.print("\033[2K");
	}

	/**
	 * Retorna os valores do histórico de perdas do modelo durante o treino.
	 * @return
	 */
	public Double[] hist() {
		return historico.toArray(new Double[]{});
	}
	
}
