import java.text.DecimalFormat;
import java.util.concurrent.TimeUnit;

import ged.Dados;
import ged.Ged;
import jnn.Funcional;
import jnn.camadas.*;
import jnn.camadas.pooling.MaxPool2D;
import jnn.core.backend.Backend;
import jnn.dataloader.DataLoader;
import jnn.dataloader.dataset.MNIST;
import jnn.io.Serializador;
import jnn.modelos.Modelo;
import jnn.modelos.Sequencial;

public class MainConv {

	/**
	 * Gerenciador de dados.
	 */
	static Ged ged = new Ged();

	/**
	 * Interface da biblioteca.
	 */
	static Funcional jnn = new Funcional();

	// controle de treino
	static final int TREINO_EPOCAS = 10;
	static final int TREINO_LOTE = 64;
	static final boolean TREINO_LOGS = true;

	// caminhos de arquivos externos
	static final String CAMINHO_SAIDA_MODELO = "./dados/modelos/modelo-treinado.nn";
	static final String CAMINHO_HISTORICO = "historico-perda.csv";

	public static void main(String[] args) {
		ged.limparConsole();

		DataLoader dlTreino = MNIST.treino();
		dlTreino.print();

		Backend.jni = true;

		Sequencial modelo = cnn();
		modelo.setHistorico(true);
		modelo.print();

		System.out.println("Treinando.");
		long tempo = System.nanoTime();
			modelo.treinar(dlTreino, TREINO_EPOCAS, TREINO_LOTE, TREINO_LOGS);
		tempo = System.nanoTime() - tempo;

		long segundosTotais = TimeUnit.NANOSECONDS.toSeconds(tempo);
		long horas = segundosTotais / 3600;
		long minutos = (segundosTotais % 3600) / 60;
		long segundos = segundosTotais % 60;

		System.out.println("\nTempo de treino: " + horas + "h " + minutos + "min " + segundos + "s");
		System.out.print("Treino -> perda: " + modelo.avaliar(dlTreino).item() + " - ");
		System.out.println("acurácia: " + formatarDecimal((modelo.avaliador().acuracia(dlTreino).item() * 100), 4) + "%");

		System.out.println("\nCarregando dados de teste.");
		DataLoader dlTeste = MNIST.teste();
		System.out.print("Teste -> perda: " + modelo.avaliar(dlTeste).item() + " - ");
		System.out.println("acurácia: " + formatarDecimal((modelo.avaliador().acuracia(dlTeste).item() * 100), 4) + "%");

		salvarModelo(modelo, CAMINHO_SAIDA_MODELO);

		exportarHistorico(modelo, CAMINHO_HISTORICO);
		executarComando("python grafico.py " + CAMINHO_HISTORICO);
	}

	/**
	 * Cria um modelo Multilayer Perceptron.
	 * @return {@code Sequencial}.
	 */
	static Sequencial mlp() {
		Sequencial modelo = new Sequencial(
			new Entrada(1, 28, 28),
			new Flatten(),
			new Densa(40, "tanh"),
			new Densa(40, "tanh"),
			new Densa(40, "tanh"),
			new Densa(10, "sigmoid")
		);

		modelo.compilar("adam", "entropia-cruzada");
		
		return modelo;
	}

	/**
	 * Cria um modelo Convoluiconal.
	 * @return {@code Sequencial}.
	 */
	static Sequencial cnn() {
		Sequencial modelo = new Sequencial(
			new Entrada(1, 28, 28),
			new Conv2D(32, new int[]{3, 3}, "relu"),
			new MaxPool2D(new int[]{2, 2}),
			new Conv2D(32, new int[]{3, 3}, "relu"),
			new MaxPool2D(new int[]{2, 2}),
			new Flatten(),
			new Densa(128, "relu"),
			new Densa(10, "softmax")
		);

		modelo.compilar("adam", "entropia-cruzada");
		
		return modelo;		
	}

	/**
	 * Salva o modelo num arquivo externo.
	 * @param modelo instância de um modelo sequencial.
	 * @param caminho caminho de destino.
	 */
	static void salvarModelo(Sequencial modelo, String caminho) {
		new Serializador().salvar(modelo, caminho);
	}

	/**
	 * Formata o valor recebido para a quantidade de casas após o ponto
	 * flutuante.
	 * @param valor valor alvo.
	 * @param casas quantidade de casas após o ponto flutuante.
	 * @return
	 */
	static String formatarDecimal(double valor, int casas) {
		String formato = "#.";
		for (int i = 0; i < casas; i++) formato += "#";

		DecimalFormat df = new DecimalFormat(formato);

		return df.format(valor);
	}

	/**
	 * Salva um arquivo csv com o historico de desempenho do modelo.
	 * @param modelo modelo.
	 * @param caminho caminho onde será salvo o arquivo.
	 */
	static void exportarHistorico(Modelo modelo, String caminho) {
		System.out.println("Exportando histórico de perda");

		double[] perdas = modelo.hist();
		double[][] dadosPerdas = new double[perdas.length][1];

		for (int i = 0; i < dadosPerdas.length; i++) {
			dadosPerdas[i][0] = perdas[i];
		}

		Dados dados = new Dados(dadosPerdas);
		ged.exportarCsv(dados, caminho);
	}

	/**
	 * Executa um comando do terminald Windows.
	 * @param comando comando para o prompt.
	 */
	static void executarComando(String comando) {
		try {
			new ProcessBuilder("cmd", "/c", comando).inheritIO().start().waitFor();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
