import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

import ged.Dados;
import ged.Ged;
import jnn.camadas.*;
import jnn.camadas.acts.ReLU;
import jnn.camadas.acts.Softmax;
import jnn.camadas.acts.Tanh;
import jnn.camadas.pooling.MaxPool2D;
import jnn.core.JNNnative;
import jnn.dataloader.DataLoader;
import jnn.dataloader.dataset.MNIST;
import jnn.io.JNNserial;
import jnn.modelos.Modelo;
import jnn.modelos.Sequencial;

public class MainConv {

	/**
	 * Gerenciador de dados.
	 */
	static Ged ged = new Ged();

	// controle de treino
	static final int TREINO_EPOCAS = 15;
	static final int TREINO_LOTE = 32;
	static final boolean TREINO_LOGS = true;

	// caminhos de arquivos externos
	static final String CAMINHO_SAIDA_MODELO = "./dados/modelos/modelo-treinado.nn";
	static final String CAMINHO_HISTORICO = "historico-perda.csv";
	
	public static void main(String[] args) {
		ged.limparConsole();
		
		DataLoader dlTreino = MNIST.treino();
		dlTreino.print();
		
		JNNnative.jni = true;

		Sequencial modelo = cnn();
		modelo.setHistorico(true);
		modelo.print();

		DataLoader dlTeste = MNIST.teste();
	
		ArrayList<Float> accs = new ArrayList<>();
		modelo.treinador().setCallback(info -> {
			float ac = modelo.avaliador().acuracia(dlTeste).item();
			accs.add(ac);
		});

		System.out.println("Treinando.");
		long tempo = System.nanoTime();
			modelo.treinar(dlTreino, TREINO_EPOCAS, TREINO_LOTE, TREINO_LOGS);
		tempo = System.nanoTime() - tempo;

		long segundosTotais = TimeUnit.NANOSECONDS.toSeconds(tempo);
		long horas = segundosTotais / 3600;
		long minutos = (segundosTotais % 3600) / 60;
		long segundos = segundosTotais % 60;

		System.out.println("\nTempo de treino: " + horas + "h " + minutos + "min " + segundos + "s");

		System.out.print("Teste -> perda: " + modelo.avaliar(dlTeste).item() + " - ");
		System.out.println("acurácia: " + formatarDecimal((modelo.avaliador().acuracia(dlTeste).item() * 100), 4) + "%");

		JNNserial.salvar(modelo, CAMINHO_SAIDA_MODELO);

		exportarHistorico(modelo, CAMINHO_HISTORICO, accs.toArray(new Float[]{}));
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
			new Densa(40),
			new Tanh(),
			new Densa(40),
			new Tanh(),
			new Densa(40),
			new Tanh(),
			new Densa(10),
			new Softmax()
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

			new Conv2D(32, new int[]{3, 3}),
			new ReLU(),
			new MaxPool2D(new int[]{2, 2}),

			new Conv2D(24, new int[]{3, 3}),
			new ReLU(),
			new MaxPool2D(new int[]{2, 2}),

			new Flatten(),

			new Densa(50),
			new ReLU(),

			new Densa(10),
			new Softmax()
		);

		modelo.compilar("adam", "entropia-cruzada");
		
		return modelo;		
	}

	/**
	 * Formata o valor recebido para a quantidade de casas após o ponto
	 * flutuante.
	 * @param valor valor alvo.
	 * @param casas quantidade de casas após o ponto flutuante.
	 * @return
	 */
	static String formatarDecimal(float valor, int casas) {
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
	static void exportarHistorico(Modelo modelo, String caminho, Float[] accs) {
		System.out.println("Exportando histórico de perda");

		Dados dados = null;

		if (accs != null) {
			float[] perdas = modelo.hist();
			float[][] dadosPerdas = new float[perdas.length][2];
			
			for (int i = 0; i < dadosPerdas.length; i++) {
				dadosPerdas[i][0] = perdas[i];
				dadosPerdas[i][1] = accs[i];
			}
			
			dados = new Dados(dadosPerdas);	

		} else {
			float[] perdas = modelo.hist();
			float[][] dadosPerdas = new float[perdas.length][1];
			
			for (int i = 0; i < dadosPerdas.length; i++) {
				dadosPerdas[i][0] = perdas[i];
			}
			
			dados = new Dados(dadosPerdas);
		}
		
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
