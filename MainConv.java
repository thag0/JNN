import java.text.DecimalFormat;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import ged.Dados;
import ged.Ged;
import jnn.Funcional;
import jnn.camadas.*;
import jnn.camadas.pooling.MaxPool2D;
import jnn.core.tensor.Tensor;
import jnn.dataloader.DataLoader;
import jnn.dataloader.dataset.MNIST;
import jnn.io.Serializador;
import jnn.io.seriais.SerialTensor;
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

	// += 1min 59s - 500 amostras - 8 epocas - 32 lote
	// += 5min 15s - 1000 amostras - 10 epocas - 64 lote
	// += 2min 12s - 1000 amostras - 10 epocas - 64 lote (paralelizando forward e backward da Conv2D)
	// dados de controle
	static final int NUM_DIGITOS_TREINO = 10;
	static final int NUM_DIGITOS_TESTE  = NUM_DIGITOS_TREINO;
	static final int NUM_AMOSTRAS_TREINO = 1_000;// max 1000
	static final int NUM_AMOSTRAS_TESTE  = 500;// max 500
	static final int TREINO_EPOCAS = 5;
	static final int TREINO_LOTE = 64;
	static final boolean TREINO_LOGS = true;

	// caminhos de arquivos externos
	static final String CAMINHO_TREINO = "./dados/mnist/treino/";
	static final String CAMINHO_TESTE = "./dados/mnist/teste/";
	static final String CAMINHO_SAIDA_MODELO = "./dados/modelos/modelo-treinado.nn";
	static final String CAMINHO_HISTORICO = "historico-perda.csv";

	public static void main(String[] args) {
		ged.limparConsole();

		DataLoader dlTreino = MNIST.treino();
		dlTreino.transformX(a -> a.div(255));//normalizar entrada entre 0 e 1

		dlTreino.print();

		Sequencial modelo = criarModelo();
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

		exportarHistorico(modelo, CAMINHO_HISTORICO);
		salvarModelo(modelo, CAMINHO_SAIDA_MODELO);
		MainImg.executarComando("python grafico.py " + CAMINHO_HISTORICO);
	}

	/*
	 * Criação de modelos para testes.
	 */
	static Sequencial criarModelo() {
		Sequencial modelo = new Sequencial(
			new Entrada(1, 28, 28),
			new Conv2D(32, new int[]{3, 3}, "relu"),
			new MaxPool2D(new int[]{2, 2}),
			new Conv2D(20, new int[]{3, 3}, "relu"),
			new MaxPool2D(new int[]{2, 2}),
			new Flatten(),
			new Densa(100, "tanh"),
			new Dropout(0.2),
			new Densa(NUM_DIGITOS_TREINO, "softmax")
		);

		// Sequencial modelo = new Sequencial(
		// 	new Entrada(1, 28, 28),
		// 	new Flatten(),
		// 	new Densa(20, "tanh"),
		// 	new Densa(20, "tanh"),
		// 	new Densa(20, "tanh"),
		// 	new Densa(NUM_DIGITOS_TREINO, "sigmoid")
		// );

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
	 * Carrega as imagens do conjunto de dados {@code MNIST}.
	 * <p>
	 *    Nota
	 * </p>
	 * O diretório deve conter subdiretórios, cada um contendo o conjunto de 
	 * imagens de cada dígito, exemplo:
	 * <pre>
	 *"mnist/treino/0"
	 *"mnist/treino/1"
	 *"mnist/treino/2"
	 *"mnist/treino/3"
	 *"mnist/treino/4"
	 *"mnist/treino/5"
	 *"mnist/treino/6"
	 *"mnist/treino/7"
	 *"mnist/treino/8"
	 *"mnist/treino/9"
	 * </pre>
	 * @param caminho caminho do diretório das imagens.
	 * @param amostras quantidade de amostras por dígito
	 * @param digitos quantidade de dígitos, iniciando do dígito 0.
	 * @return dados carregados.
	 */
	static Tensor[] carregarAmostrasMNIST(String caminho, int amostras, int digitos) {
		final Tensor[] arr = new Tensor[digitos * amostras];
		final int numThreads = Runtime.getRuntime().availableProcessors();
		SerialTensor st = new SerialTensor();

		try (ExecutorService exec = Executors.newFixedThreadPool(numThreads)) {
			int id = 0;
			for (int digito = 0; digito < digitos; digito++) {
				for (int amostra = 0; amostra < amostras; amostra++) {
					final String caminhoCompleto = caminho + digito + "/img_" + amostra + ".jpg";
					final int indice = id;
					
					exec.submit(() -> {
						try {
							Tensor img = st.lerImagem(caminhoCompleto);
							arr[indice] = img.unsqueeze(0); // ser 3d
						} catch (Exception e) {
							System.out.println(e.getMessage());
							System.exit(1);
						}
					});

					id++;
				}
			}
  
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
  
		System.out.println("Imagens carregadas (" + arr.length + ").");
  
		return arr;		
	}

	/**
	 * Gera os rótulos do conjunto de dados {@code MNIST}.
	 * @param amostras quantidades de amostras por dítigo.
	 * @param digitos quantidade de dítigos, começando do 0.
	 * @return dados carregados.
	 */
	static Tensor[] criarRotulosMNIST(int amostras, int digitos) {
		Tensor[] rotulos = new Tensor[digitos * amostras]; 

		for (int digito = 0; digito < digitos; digito++) {
			for (int amostra = 0; amostra < amostras; amostra++) {
				double[] data = new double[digitos];
				data[digito] = 1;
				
				int indice = digito * amostras + amostra;
				rotulos[indice] = new Tensor(data);
			}
		}

		return rotulos;
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
}
