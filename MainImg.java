import java.awt.image.BufferedImage;
import java.text.DecimalFormat;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;

import ged.Dados;
import ged.Ged;
import geim.Geim;
import geim.imagem.Imagem;
import jnn.JNN;
import jnn.camadas.Densa;
import jnn.camadas.Entrada;
import jnn.camadas.acts.Sigmoid;
import jnn.camadas.acts.Tanh;
import jnn.core.parallel.PoolFactory;
import jnn.core.tensor.Tensor;
import jnn.dataloader.DataLoader;
import jnn.modelos.Modelo;
import jnn.modelos.Sequencial;
import jnn.treino.scheduler.StepLR;
import jnnview.JNNwindow;

public class MainImg {
	static Ged ged = new Ged();
	static Geim geim = new Geim();

	static final int EPOCAS = 5 * 1000;
	static final int ESCALA_RENDER = 10;
	static final boolean historico = true;
	static final String CAMINHO_HISTORICO = "historico-perda.csv";
	static final String CAMINHO_IMAGEM = "./dados/mnist/img_0.png";

	public static void main(String[] args) {      
		ged.limparConsole();

		BufferedImage imagem = geim.lerImagem(CAMINHO_IMAGEM);
		int tamEntrada = 2;
		int tamSaida = 1;
		
		float[][] dados;
		if 		(tamSaida == 1) dados = imagemParaDadosTreinoEscalaCinza(imagem);
		else if (tamSaida == 3) dados = imagemParaDadosTreinoRGB(imagem);
		else throw new IllegalArgumentException("\nImagem deve ser em Escala de Cinza ou RGB");

		float[][] in  = (float[][]) ged.separarDadosEntrada(dados, tamEntrada);
		float[][] out = (float[][]) ged.separarDadosSaida(dados, tamSaida);

		Tensor[] x = JNN.arrayParaTensores(in);
		Tensor[] y = JNN.arrayParaTensores(out);
		DataLoader dl = new DataLoader(x, y);
		dl.aplicarY(t -> t.div(255));// normalizar saída de 0-255 para 0-1
		dl.print();

		Modelo modelo = modelo(tamEntrada, tamSaida);
		modelo.print();

		System.out.println("Treinando.");
		long tempo = treinoEmPainel(modelo, dl, imagem.getWidth(), imagem.getHeight());
		long hrs = TimeUnit.NANOSECONDS.toHours(tempo);
		long min = TimeUnit.NANOSECONDS.toMinutes(tempo);
		long sec = TimeUnit.NANOSECONDS.toSeconds(tempo);

		System.out.println("Perda = " + modelo.avaliar(dl).item());
		System.out.println("Tempo de treinamento: " + hrs + "h " + min + "m " + sec + "s");

		if (historico) {
			exportarHistorico(modelo, CAMINHO_HISTORICO);
			executarComando("python grafico.py " + CAMINHO_HISTORICO);
		}

		exportarImagemEscalaCinza(modelo, 28, 28, 10, "img-ampliada.png");
	}

	/**
	 * Cria um modelo Sequencial.
	 * @param in quantidade de dados de entrada.
	 * @param out quantidade de dados de saída.
	 * @return {@code Modelo} criado.
	 */
	static Modelo modelo(int in, int out) {
		Sequencial modelo = new Sequencial(
			new Entrada(in),
			new Densa(14),
			new Tanh(),
			new Densa(14),
			new Tanh(),
			new Densa(out),
			new Sigmoid()
		);

		Object optm = "sgd";
		Object loss = "mse"; 

		modelo.compilar(optm, loss);
		modelo.setHistorico(historico);

		modelo.treinador().setScheduler(new StepLR(modelo.otm(), 25, 0.98f));

		return modelo;
	}

	/**
	 * Treina e exibe o resultado da Rede Neural no painel.
	 * @param modelo modelo de rede neural usado no treino.
	 * @param loader {@code DataLoader} com dados de treino.
	 * @param altura altura da janela renderizada.
	 * @param largura largura da janela renderizada.
	 * @return tempo (em nano segundos) do treino.
	 */
	static long treinoEmPainel(Modelo modelo, DataLoader loader, int altura, int largura) {
		final int FPS = 60_000000;
		final int EPOCAS_POR_FRAME = 65;

		Tensor in = new Tensor(2);
		Tensor img = new Tensor(altura, largura);
		JNNwindow window = new JNNwindow((int) ESCALA_RENDER);
		window.atualizar(img.array(), img.shape());
		
		window.exibir("Titulo");
		
		//trabalhar com o tempo de renderização baseado no fps
		double intervaloDesenho = 1_000_000_000/FPS;
		double proximoTempoDesenho = System.nanoTime() + intervaloDesenho;
		double tempoRestante;

		int i = 0;
		long tempoTreino = System.nanoTime();
		while (i < EPOCAS && window.isVisible()) {
			modelo.treinar(loader, EPOCAS_POR_FRAME, false);
			gerarImagem(modelo, in, img);
			window.atualizar(img.array(), img.shape());
			window.exibir(i + "/" + EPOCAS);
			
			i += EPOCAS_POR_FRAME;

			try {
				tempoRestante = proximoTempoDesenho - System.nanoTime();
				tempoRestante /= 1_000_000;
				if (tempoRestante < 0) tempoRestante = 0;

				Thread.sleep((long)tempoRestante);
				proximoTempoDesenho += intervaloDesenho;

			} catch (Exception e) {}
		}

		tempoTreino = System.nanoTime() - tempoTreino;
		
		return tempoTreino;
	}

	/**
	 * Atualiza a imagem para o render.
	 * @param modelo modelo base.
	 * @param in tensor cache de entrada.
	 * @param img imagem final para desenho.
	 */
	static void gerarImagem(Modelo modelo, Tensor in, Tensor img) {
		float[] dataImg = img.array();
		int alt = img.tamDim(0);
		int larg = img.tamDim(1);

		float[] dataIn = in.array();

		for (int y = 0; y < alt; y++) {
			dataIn[1] = (float) y / (alt-1);

			for (int x = 0; x < larg; x++) {
				dataIn[0] = (float) x / (larg - 1);		
				float b = modelo.forward(in).item();
				dataImg[y * larg + x] = b;
			}
		}
	}

	/**
	 * Salva um arquivo csv com o historico de desempenho do modelo.
	 * @param modelo modelo.
	 * @param caminho caminho onde será salvo o arquivo.
	 */
	static void exportarHistorico(Modelo modelo, String caminho) {
		System.out.println("Exportando histórico de perda");
		float[] perdas = modelo.hist();
		float[][] valores = new float[perdas.length][1];

		final int t = Runtime.getRuntime().availableProcessors();
		try (ForkJoinPool pool = PoolFactory.pool(t)) {
			for (int i = 0, n = valores.length; i < n; i++) {
				final int id = i;
				pool.submit(() -> {
					valores[id][0] = perdas[id];
				});
			}
		}

		Dados dados = new Dados(valores);
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

	/**
	 * Utiliza o modelo para criar e exportar uma imagem.
	 * @param modelo modelo para uso.
	 * @param altBase altura base da imagem.
	 * @param largBase largura base da imagem.
	 * @param escala fator de escala para altura e largura final.
	 * @param caminho caminho para o arquivo.
	 */
	static void exportarImagemEscalaCinza(Modelo modelo, int altBase, int largBase, double escala, String caminho) {
		modelo.loteZero();

		if (escala <= 0) throw new IllegalArgumentException("O valor de escala não pode ser menor que 1.");
		if (modelo.camadaSaida().tamSaida() != 1) {
			throw new IllegalArgumentException(
				"O modelo deve trabalhar apenas com uma unidade na camada de saída para a escala de cinza."
			);
		}

		int larguraFinal = (int)(largBase * escala);
		int alturaFinal = (int)(altBase * escala);
		Imagem imagemAmpliada = geim.gerarEstruturaImagem(larguraFinal, alturaFinal);
		
		int alturaImagem = imagemAmpliada.altura();
		int larguraImagem = imagemAmpliada.largura();

		//gerenciar multithread
		int numThreads = Runtime.getRuntime().availableProcessors();
		if (numThreads > 1) {
			//só pra não sobrecarregar o processador
			numThreads = (int)(numThreads / 2);
		}

		Thread[] threads = new Thread[numThreads];
		Modelo[] clones = new Modelo[numThreads];
		for (int i = 0; i < numThreads; i++) {
			clones[i] = modelo.clone();
		}

		int alturaPorThead = alturaImagem / numThreads;

		for (int i = 0; i < numThreads; i++) {
			final int id = i;
			final int inicio = i * alturaPorThead;
			final int fim = inicio + alturaPorThead;
			
			threads[i] = new Thread(() -> {
				Tensor in = new Tensor(2);

				for (int y = inicio; y < fim; y++) {
					for (int x = 0; x < larguraImagem; x++) {
						in.set(((float) x / (larguraImagem-1)), 0);
						in.set(((float) y / (alturaImagem-1)), 1);
						float[] saida = new float[1];
					
						clones[id].forward(in);
					
						saida[0] = clones[id].saidaParaArray()[0] * 255f;

						synchronized(imagemAmpliada) {
							imagemAmpliada.set(x, y, (int)saida[0], (int)saida[0], (int)saida[0]);
						}
					}
				}
			});

			threads[i].start();
		}

		try {
			for (Thread thread : threads) {
				thread.join();
			}
		} catch(Exception e) {
			System.out.println("Ocorreu um erro ao tentar salvar a imagem.");
			e.printStackTrace();
		}

		imagemAmpliada.paraPNG(caminho);
	}

	/**
	 * Utiliza o modelo para criar e exportar uma imagem.
	 * @param modelo modelo para uso.
	 * @param altBase altura base da imagem.
	 * @param largBase largura base da imagem.
	 * @param escala fator de escala para altura e largura final.
	 * @param caminho caminho para o arquivo.
	 */
	static void exportarImagemRGB(Modelo modelo, int altBase, int largBase, double escala, String caminho) {
		if (escala <= 0) throw new IllegalArgumentException("O valor de escala não pode ser menor que 1.");
		if (modelo.camadaSaida().tamSaida() != 3) {
			throw new IllegalArgumentException(
				"O modelo deve trabalhar apenas com três neurônios na saída para RGB."
			);
		}

		//estrutura de dados da imagem
		int larguraFinal = (int)(largBase * escala);
		int alturaFinal = (int)(altBase * escala);
		Imagem imagemAmpliada = geim.gerarEstruturaImagem(larguraFinal, alturaFinal);

		int alturaImagem = imagemAmpliada.altura();
		int larguraImagem = imagemAmpliada.largura();

		//gerenciar multithread
		int numThreads = Runtime.getRuntime().availableProcessors();
		if (numThreads > 1) {
			//só pra não sobrecarregar o processador
			numThreads = (int)(numThreads / 2);
		}

		Thread[] threads = new Thread[numThreads];
		Modelo[] redes = new Modelo[numThreads];
		for (int i = 0; i < numThreads; i++) {
			redes[i] = modelo.clone();
		}

		int alturaPorThead = alturaImagem / numThreads;

		for (int i = 0; i < numThreads; i++) {
			final int id = i;
			final int inicio = i * alturaPorThead;
			final int fim = inicio + alturaPorThead;
			
			threads[i] = new Thread(() -> {
				Tensor in = new Tensor(2);

				for (int y = inicio; y < fim; y++) {
					for (int x = 0; x < larguraImagem; x++) {
						float[] entrada = new float[2];
						in.set(((float) x / (larguraImagem-1)), 0);
						in.set(((float) y / (alturaImagem-1)), 1);
						float[] saida = new float[3];

						entrada[0] = (float)x / (larguraImagem-1);
						entrada[1] = (float)y / (alturaImagem-1);
					
						redes[id].forward(in);
						float[] s = redes[id].saidaParaArray();
					
						saida[0] = s[0] * 255f;
						saida[1] = s[1] * 255f;
						saida[2] = s[2] * 255f;

						synchronized (imagemAmpliada) {
							imagemAmpliada.set(x, y, (int)saida[0], (int)saida[1], (int)saida[2]);
						}
					}
				}
			});

			threads[i].start();
		}

		try {
			for (Thread thread : threads) {
				thread.join();
			}
		} catch (Exception e) {
			System.out.println("Ocorreu um erro ao tentar salvar a imagem.");
			e.printStackTrace();
		}

		imagemAmpliada.paraPNG(caminho);
	}

	/**
	 * Gera um conjunto de dados de treino baseado na imagem.
	 * @param img imagem base.
	 * @return dados de treino.
	 */
	static float[][] imagemParaDadosTreinoEscalaCinza(BufferedImage imagem) {
		if (imagem == null) throw new IllegalArgumentException("A imagem fornecida é nula.");

		int larguraImagem = imagem.getWidth();
		int alturaImagem = imagem.getHeight();

		float[][] dadosImagem = new float[larguraImagem * alturaImagem][3];
		int[][] vermelho = geim.getR(imagem);
		int[][] verde = geim.getG(imagem);
		int[][] azul = geim.getB(imagem);

		int contador = 0;
		for (int y = 0; y < alturaImagem; y++) {
			for (int x = 0; x < larguraImagem; x++) {
				int r = vermelho[y][x];
				int g = verde[y][x];
				int b = azul[y][x];

				// preenchendo os dados na matriz
				float xNormalizado = (float) x / (larguraImagem - 1);
				float yNormalizado = (float) y / (alturaImagem - 1);
				float escalaCinza = (r + g + b) / 3.0f;
				
				dadosImagem[contador][0] = xNormalizado;// x
				dadosImagem[contador][1] = yNormalizado;// y
				dadosImagem[contador][2] = escalaCinza;// escala de cinza

				contador++;
			}
		}
		

		return dadosImagem;
	}

	/**
	 * Gera um conjunto de dados de treino baseado na imagem.
	 * @param img imagem base.
	 * @return dados de treino.
	 */
	static float[][] imagemParaDadosTreinoRGB(BufferedImage imagem) {
		if (imagem == null) throw new IllegalArgumentException("A imagem fornecida é nula.");
		int larguraImagem = imagem.getWidth();
		int alturaImagem = imagem.getHeight();

		float[][] dadosImagem = new float[larguraImagem * alturaImagem][5];
		int[][] vermelho = geim.getR(imagem);
		int[][] verde = geim.getG(imagem);
		int[][] azul = geim.getB(imagem);

		int contador = 0;

		for (int y = 0; y < alturaImagem; y++) {
			for (int x = 0; x < larguraImagem; x++) { 
				int r = vermelho[y][x];
				int g = verde[y][x];
				int b = azul[y][x];

				// preenchendo os dados na matriz
				float xNormalizado = (float) x / (larguraImagem-1);
				float yNormalizado = (float) y / (alturaImagem-1);  
				float rNormalizado = (float) r / 255f;
				float gNormalizado = (float) g / 255f;
				float bNormalizado = (float) b / 255f;

				dadosImagem[contador][0] =  xNormalizado;// x
				dadosImagem[contador][1] =  yNormalizado;// y
				dadosImagem[contador][2] = rNormalizado;// vermelho
				dadosImagem[contador][3] = gNormalizado;// verde
				dadosImagem[contador][4] = bNormalizado;// azul

				contador++;
			}
		}

		return dadosImagem;
	}

}