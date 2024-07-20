package jnn.treinamento;

import java.util.Random;

import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;

/**
 * Classe auxiliar no treinamento, faz uso de ferramentas que podem
 * ser compartilhadas entre os diferentes tipos de modelos de treinamento.
 */
public class AuxTreino {

	/**
	 * Gerador de números aleatórios.
	 */
	Random random = new Random();

	/**
	 * Configura a seed inicial do gerador de números aleatórios.
	 * @param seed nova seed.
	 */
	public void setSeed(long seed) {
		random.setSeed(seed);
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
	 * @param camadas conjunto de camadas de um modelo.
	 * @param perda função de perda configurada para o modelo.
	 * @param prev {@code Tensor} contendos os dados previstos.
	 * @param real {@code Tensor} contendos os dados reais (rotulados).
	 */
	public void backpropagation(Modelo modelo, Tensor prev, Tensor real) {
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

	public void limparLinha() {
		System.out.print("\033[2K");
	}
}
