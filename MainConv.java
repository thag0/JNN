import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

import ged.Dados;
import ged.Ged;
import jnn.camadas.*;
import jnn.camadas.acts.ReLU;
import jnn.camadas.acts.Softmax;
import jnn.camadas.acts.Tanh;
import jnn.camadas.pooling.GlobalAvgPool2D;
import jnn.camadas.pooling.MaxPool2D;
import jnn.core.JNNnative;
import jnn.dataloader.DataLoader;
import jnn.dataloader.dataset.CIFAR10;
import jnn.dataloader.transform.Norm;
import jnn.dataloader.transform.Transform;
import jnn.io.JNNserial;
import jnn.modelos.Modelo;
import jnn.modelos.Sequencial;

public class MainConv {

	/**
	 * Gerenciador de dados.
	 */
	static Ged ged = new Ged();

	// controle de treino
	static final int TREINO_EPOCAS = 100;
	static final int TREINO_LOTE = 64;
	static final boolean TREINO_LOGS = true;

	// caminhos de arquivos externos
	static final String CAMINHO_SAIDA_MODELO = "./dados/modelos/modelo-treinado.nn";
	static final String CAMINHO_HISTORICO = "historico-perda.csv";
	
	public static void main(String[] args) {
		ged.limparConsole();
		
		Transform norm = new Norm(
			new float[]{0.4914f, 0.4822f, 0.4465f},
			new float[]{0.247f,  0.243f,  0.261f}
		);//peguei de um forum do pytorch
		
		final DataLoader treino = CIFAR10.treino().aplicarX(norm);
		treino.print();
				
		final DataLoader teste = CIFAR10.teste().aplicarX(norm);

		Sequencial modelo = cnn();
		modelo.setHistorico(true);
		modelo.print();
		
		var accs = new ArrayList<Float>();
		modelo.treinador().setCallback(info -> {
			float ac = modelo.avaliador().acuracia(teste).item();
			accs.add(ac);
		});
		
		System.out.println("Treinando.");
		JNNnative.jni = true;
		long tempo = System.nanoTime();
			modelo.treinar(treino, TREINO_EPOCAS, TREINO_LOTE, TREINO_LOGS);
		tempo = System.nanoTime() - tempo;

		long segundosTotais = TimeUnit.NANOSECONDS.toSeconds(tempo);
		long horas = segundosTotais / 3600;
		long minutos = (segundosTotais % 3600) / 60;
		long segundos = segundosTotais % 60;

		System.out.println("\nTempo de treino: " + horas + "h " + minutos + "min " + segundos + "s");

		System.out.println("loss: " + modelo.avaliar(teste));
		System.out.println("acc: " + modelo.avaliador().acuracia(teste));

		JNNserial.salvar(modelo, CAMINHO_SAIDA_MODELO);

		exportarHistorico(modelo, CAMINHO_HISTORICO, accs);
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
		int[] convK = {3, 3};
		int[] poolK = {2, 2};

		Sequencial modelo = new Sequencial(
			new Entrada(3, 32, 32),

			new Conv2D(32, convK, "same", "he"),
			new ReLU(),
			new Conv2D(32, convK, "same", "he"),
			new ReLU(),
			new MaxPool2D(poolK),
			
			new Conv2D(64, convK, "same", "he"),
			new ReLU(),
			new Conv2D(64, convK, "same", "he"),
			new ReLU(),
			new MaxPool2D(poolK),
			
			new Conv2D(128, convK, "same", "he"),
			new ReLU(),
			new Conv2D(128, convK, "same", "he"),
			new ReLU(),
			new MaxPool2D(poolK),

			new GlobalAvgPool2D(),

			new Dropout(0.5),
			new Densa(128, "he"),
			new ReLU(),

			new Densa(10, "he"),
			new Softmax()
		);

		modelo.compilar("adam", "entropia-cruzada");
		
		return modelo;		
	}

	/**
	 * Salva um arquivo csv com o historico de desempenho do modelo.
	 * @param modelo modelo.
	 * @param caminho caminho onde será salvo o arquivo.
	 */
	static void exportarHistorico(Modelo modelo, String caminho, ArrayList<Float> accs) {
		System.out.println("Exportando histórico de perda");

		Dados dados = null;

		if (accs != null) {
			float[] perdas = modelo.hist();
			float[][] dadosPerdas = new float[perdas.length][2];
			
			for (int i = 0; i < dadosPerdas.length; i++) {
				dadosPerdas[i][0] = perdas[i];
				dadosPerdas[i][1] = accs.get(i);
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
