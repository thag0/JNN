import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

import ged.Dados;
import ged.Ged;
import jnn.core.JNNnative;
import jnn.dataloader.DataLoader;
import jnn.dataloader.dataset.CIFAR10;
import jnn.dataloader.dataset.MNIST;
import jnn.dataloader.transform.ColorJitter;
import jnn.dataloader.transform.Compose;
import jnn.dataloader.transform.CutOut;
import jnn.dataloader.transform.HFlip;
import jnn.dataloader.transform.LabelSmoothing;
import jnn.dataloader.transform.Norm;
import jnn.dataloader.transform.RandomCrop;
import jnn.dataloader.transform.Transform;
import jnn.io.JNNserial;
import jnn.layers.*;
import jnn.layers.acts.ReLU;
import jnn.layers.acts.Softmax;
import jnn.layers.acts.Tanh;
import jnn.layers.pooling.GlobalAvgPool2D;
import jnn.layers.pooling.MaxPool2D;
import jnn.models.Model;
import jnn.models.Sequential;
import jnn.otm.SGD;
import jnn.otm.scheduler.CosineAnnealingLR;

public class MainConv {

	/**
	 * Gerenciador de dados.
	 */
	static Ged ged = new Ged();

	// controle de treino
	static final int TREINO_EPOCAS = 30;
	static final int TREINO_LOTE = 64;
	static final boolean TREINO_LOGS = true;

	// caminhos de arquivos externos
	static final String CAMINHO_SAIDA_MODELO = "./dados/modelos/modelo-treinado.nn";
	static final String CAMINHO_HISTORICO = "historico-perda.csv";
	
	public static void main(String[] args) {
		ged.limparConsole();
		
		DataLoader[] data = getData("mnist");
		final DataLoader treino = data[0];
		final DataLoader teste  = data[1];

		treino.print();

		Sequential model = cnnMNIST();
		model.setHistorico(true);
		model.print();
		
		var accs = new ArrayList<Float>();
		model.trainer().setCallback(info -> {
			model.eval();
				float ac = model.evaluator().accuracy(teste).item();
			model.train();
			
			accs.add(ac);
			System.out.println(" acurácia: " + (ac*100) + "%");
		});

		model.trainer().setScheduler(
			new CosineAnnealingLR(
				model.optim(),
				0.0005f,
				model.optim().getLr(),
				(TREINO_EPOCAS * (treino.len() / TREINO_LOTE))
			)
		);
		
		System.out.println("Treinando.");
		JNNnative.on();
		JNNnative.setArenaSize(10 * 1024 * 1024);
		long tempo = System.nanoTime();
			model.train();
			model.fit(treino, TREINO_EPOCAS, TREINO_LOTE, TREINO_LOGS);
		tempo = System.nanoTime() - tempo;

		long segundosTotais = TimeUnit.NANOSECONDS.toSeconds(tempo);
		long horas = segundosTotais / 3600;
		long minutos = (segundosTotais % 3600) / 60;
		long segundos = segundosTotais % 60;

		System.out.println("\nTempo de treino: " + horas + "h " + minutos + "min " + segundos + "s");

		model.eval();
		System.out.println("loss: " + model.computeLoss(teste));
		System.out.println("acc: "  + model.evaluator().accuracy(teste));

		JNNserial.salvar(model, CAMINHO_SAIDA_MODELO);

		exportarHistorico(model, CAMINHO_HISTORICO, accs);
		executarComando("python grafico.py " + CAMINHO_HISTORICO);
	}

	static DataLoader[] getData(String dataset) {
		DataLoader train, test = null;

		switch (dataset.toLowerCase().trim()) {
			case "mnist":	
				train = MNIST.train();
				test  = MNIST.test();
			break;

			case "cifar10":
				Transform norm = new Norm(
					new float[]{0.4914f, 0.4822f, 0.4465f},
					new float[]{0.247f,  0.243f,  0.261f}
				);//peguei de um forum do pytorch
				
				train = CIFAR10.train();
				train.setTransformX(new Compose(
					new ColorJitter(.2, .2, .2),
					new HFlip(0.5),
					new RandomCrop(32, 32, 4, true),
					norm,
					new CutOut(8)
				));
					
				test = CIFAR10.test().applyX(norm);
				test.applyY(new LabelSmoothing(1e-1));
			break;

			default:
				throw new IllegalArgumentException("dataset \'" + dataset +"\' inválido");
		}

		return new DataLoader[] {
			train,
			test
		};
	}

	/**
	 * Cria um modelo Multilayer Perceptron.
	 * @return {@code Sequencial}.
	 */
	static Sequential mlp() {
		Sequential model = new Sequential(
			new Input(1, 28, 28),
			new Flatten(),
			new Dense(40),
			new Tanh(),
			new Dense(40),
			new Tanh(),
			new Dense(40),
			new Tanh(),
			new Dense(10),
			new Softmax()
		);

		model.compile("adam", "entropia-cruzada");
		
		return model;
	}

	/**
	 * Cria um modelo Convoluiconal.
	 * @return {@code Sequencial}.
	 */
	static Sequential cnnCifar10() {
		int[] convK = {3, 3};
		int[] poolK = {2, 2};

		Sequential model = new Sequential(
			new Input(3, 32, 32),

			new Conv2D(32, convK, "same", "he"),
			new BatchNorm2D(),
			new ReLU(),
			new Conv2D(32, convK, "same", "he"),
			new BatchNorm2D(),
			new ReLU(),
			new MaxPool2D(poolK),
			
			new Conv2D(64, convK, "same", "he"),
			new BatchNorm2D(),
			new ReLU(),
			new Conv2D(64, convK, "same", "he"),
			new BatchNorm2D(),
			new ReLU(),
			new MaxPool2D(poolK),
			
			new Conv2D(128, convK, "same", "he"),
			new BatchNorm2D(),
			new ReLU(),
			new Conv2D(128, convK, "same", "he"),
			new BatchNorm2D(),
			new ReLU(),
			new MaxPool2D(poolK),

			new GlobalAvgPool2D(),

			new Dense(10, "glorot-uniforme"),
			new Softmax()
		);

		for (Layer layer : model) if (layer instanceof Conv2D) layer.setBias(false);
		
		Object otm = new SGD(0.01, 0.9, true);
		Object loss = "cross-entropy";

		model.compile(otm, loss);

		return model;		
	}

	static Sequential cnnMNIST() {
		int[] convK = {3, 3};
		int[] poolK = {2, 2};

		Sequential model = new Sequential(
			new Input(1, 28, 28),

			new Conv2D(16, convK),
			new ReLU(),
			new MaxPool2D(poolK),
			
			new Conv2D(32, convK),
			new ReLU(),
			new MaxPool2D(poolK),

			new GlobalAvgPool2D(),

			new Dropout(0.3),
			new Dense(100, "he"),
			new ReLU(),

			new Dense(10, "glorot-uniforme"),
			new Softmax()
		);

		Object otm = "adam";
		Object loss = "cross-entropy";
		
		model.compile(otm, loss);

		return model;		
	}

	/**
	 * Salva um arquivo csv com o historico de desempenho do modelo.
	 * @param modelo modelo.
	 * @param caminho caminho onde será salvo o arquivo.
	 */
	static void exportarHistorico(Model modelo, String caminho, ArrayList<Float> accs) {
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
