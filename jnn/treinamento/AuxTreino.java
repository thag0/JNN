package jnn.treinamento;

import java.util.Random;

import jnn.avaliacao.perda.Perda;
import jnn.camadas.Camada;
import jnn.core.OpArray;
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
	 * Operador para arrays.
	 */
	OpArray oparr = new OpArray();

	/**
	 * Configura a seed inicial do gerador de números aleatórios.
	 * @param seed nova seed.
	 */
	public void setSeed(long seed) {
		random.setSeed(seed);
	}

	/**
	 * Realiza a retropropagação de gradientes de cada camada para a atualização de pesos.
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
	 * Embaralha os dados da matriz usando o algoritmo Fisher-Yates.
	 * @param entradas matriz com os dados de entrada.
	 * @param saidas matriz com os dados de saída.
	 */
	public void embaralharDados(Tensor[] entradas, Tensor[] saidas) {
		int linhas = entradas.length;
		int i, idAleatorio;

		Tensor temp;
		for (i = linhas - 1; i > 0; i--) {
			idAleatorio = random.nextInt(i+1);

			//trocar entradas
			temp = entradas[i];
			entradas[i] = entradas[idAleatorio];
			entradas[idAleatorio] = temp;

			//trocar saídas
			temp = saidas[i];
			saidas[i] = saidas[idAleatorio];
			saidas[idAleatorio] = temp;
		}
	}

	/**
	 * Dedicado para treino em lote e multithread em implementações futuras.
	 * @param dados conjunto de dados completo.
	 * @param inicio índice de inicio do lote.
	 * @param fim índice final do lote.
	 * @return lote contendo os dados de acordo com os índices fornecidos.
	 */
	public Tensor[] obterSubMatriz(Tensor[] dados, int inicio, int fim) {
		if (inicio < 0 || fim > dados.length || inicio >= fim) {
			throw new IllegalArgumentException("Índices de início ou fim inválidos.");
		}

		int linhas = fim - inicio;
		Tensor[] subMatriz = new Tensor[linhas];

		System.arraycopy(dados, inicio, subMatriz, 0, linhas);

		return subMatriz;
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
