package jnn.treinamento;

import java.lang.reflect.Array;

import java.util.Random;

import jnn.avaliacao.perda.Perda;
import jnn.camadas.Camada;
import jnn.core.tensor.Tensor;

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
	public void backpropagation(Camada[] camadas, Perda perda, Tensor prev, Tensor real) {
		Tensor grad = perda.derivada(prev, real);
		for (int i = camadas.length-1; i >= 0; i--) {
			grad = camadas[i].backward(grad);
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
	 * Separa um sub conjunto dos dados do array de acordo com os índices fornecidos.
	 * @param <T> tipo de dados do array.
	 * @param arr conjunto de dados desejado.
	 * @param inicio indice inicial (inclusivo).
	 * @param fim índice final (exclusivo).
	 * @return sub conjunto dos dados fornecidos.
	 */
    public <T> T[] subArray(T[] arr, int inicio, int fim) {
        if (inicio < 0 || fim > arr.length || inicio >= fim) {
            throw new IllegalArgumentException("Índices de início ou fim inválidos.");
        }

        int tamanho = fim - inicio;

        @SuppressWarnings("unchecked")
        T[] subArr = (T[]) Array.newInstance(arr.getClass().getComponentType(), tamanho);
        System.arraycopy(arr, inicio, subArr, 0, tamanho);

        return subArr;
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
