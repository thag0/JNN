package jnn.treinamento;

import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;

/**
 * Interface para usar os métodos de treino sequencial e treino em
 * lote dos modelos.
 */
public class Treinador {

	/**
	 * Operador do treino sequencial.
	 */
	Treino treino;

	/**
	 * Operador do treino em lotes.
	 */
	TreinoLote treinoLote;

	/**
	 * Responsável por organizar os tipos dos modelos.
	 */
	public Treinador() {
		treino 	   = new Treino(false);
		treinoLote = new TreinoLote(false);
	}

	/**
	 * Configura a seed inicial do gerador de números aleatórios.
	 * @param seed nova seed.
	 */
	public void setSeed(long seed) {
		treino.setSeed(seed);
		treinoLote.setSeed(seed);
	}

	/**
	 * Configura o cálculo para o histórico de perdas durante o treinamento.
	 * @param calcular calcular ou não o histórico de custo.
	 */
	public void setHistorico(boolean calcular) {
		treino.setHistorico(calcular);
		treinoLote.setHistorico(calcular);
	}

	/**
	 * Treina o modelo ajustando seus parâmetros treináveis usando
	 * os dados fornecidos.
	 * @param modelo modelo que será treinada.
	 * @param entradas {@code Tensores} contendos os dados de entrada.
	 * @param saidas {@code Tensores} contendos os dados de saída (rótulos).
	 * @param epochs quantidade de épocas de treinamento.
	 * @param logs logs para perda durante as épocas de treinamento.
	 */
	public void treino(Modelo modelo, Tensor[] entradas, Tensor[] saidas, int epochs, boolean logs) {
		executar(modelo, entradas, saidas, epochs, 0, logs);
	}

	/**
	 * Treina a rede neural calculando os erros dos neuronios, seus gradientes para cada peso e 
	 * passando essas informações para o otimizador configurado ajustar os pesos.
	 * @param modelo rede neural que será treinada.
	 * @param entradas {@code Tensores} contendos os dados de entrada.
	 * @param saidas {@code Tensores} contendos os dados de saída (rótulos).
	 * @param epochs quantidade de épocas de treinamento.
	 * @param tamLote tamanho do lote.
	 * @param logs logs para perda durante as épocas de treinamento.
	 */
	public void treino(Modelo modelo, Tensor[] entradas, Tensor[] saidas, int epochs, int tamLote, boolean logs) {
		executar(modelo, entradas, saidas, epochs, tamLote, logs);
	}

	/**
	 * Executa a função de treino de acordo com os valores configurados.
	 * @param modelo rede neural que será treinada.
	 * @param epochs quantidade de épocas de treinamento.
	 * @param tamLote tamanho do lote.
	 * @param logs logs para perda durante as épocas de treinamento.
	 */
	private void executar(Modelo modelo, Tensor[] entradas, Tensor[] saidas, int epochs, int tamLote, boolean logs) {
		modelo.treino(true);

		if (tamLote > 1) {
			treinoLote.treinar(modelo, entradas, saidas, epochs, tamLote, logs);
			treinoLote.ultimoUsado = true;
		
		} else {
			treino.treinar(modelo, entradas, saidas, epochs, logs);
			treino.ultimoUsado = true;
		}

		treino.ultimoUsado = treinoLote.ultimoUsado ? false : true;
		
		modelo.treino(false);
	}

	/**
	 * Retorna uma lista contendo os valores de custo da rede
	 * a cada época de treinamento.
	 * @return lista com os custo por época durante a fase de treinamento.
	 */
	public double[] historico() {
		Object[] hist = treino.ultimoUsado ? treino.historico() : treinoLote.historico();
		double[] h = new double[hist.length];

		for (int i = 0; i < h.length; i++) {
			h[i] = (double) hist[i];
		}

		return h;
	}
	
}
