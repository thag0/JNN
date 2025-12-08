import java.awt.image.BufferedImage;
import java.text.DecimalFormat;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import ged.Dados;
import ged.Ged;
import geim.Geim;
import geim.imagem.Imagem;
import jnn.Funcional;
import jnn.camadas.Densa;
import jnn.camadas.Entrada;
import jnn.core.tensor.Tensor;
import jnn.dataloader.DataLoader;
import jnn.modelos.Modelo;
import jnn.modelos.RedeNeural;
import jnn.modelos.Sequencial;
import jnn.otm.*;
import render.JanelaTreino;

public class MainImg {
	static Ged ged = new Ged();
	static Geim geim = new Geim();
	static Funcional jnn = new Funcional();

	static final int EPOCAS = 10 * 1000;
	static final double ESCALA_RENDER = 10;
	static final boolean historico = true;
	static final String CAMINHO_HISTORICO = "historico-perda.csv";
	static final String CAMINHO_IMAGEM = "./dados/mnist/img_0.png";
	// static final String caminhoImagem = "./dados/32x32/circulos.png";

	public static void main(String[] args) {      
		ged.limparConsole();

		BufferedImage imagem = geim.lerImagem(CAMINHO_IMAGEM);
		int tamEntrada = 2;
		int tamSaida = 1;
		
		double[][] dados;
		if 		(tamSaida == 1) dados = imagemParaDadosTreinoEscalaCinza(imagem);
		else if (tamSaida == 3) dados = imagemParaDadosTreinoRGB(imagem);
		else throw new IllegalArgumentException("\nImagem deve ser em Escala de Cinza ou RGB");

		double[][] in  = (double[][]) ged.separarDadosEntrada(dados, tamEntrada);
		double[][] out = (double[][]) ged.separarDadosSaida(dados, tamSaida);

		Tensor[] x = jnn.arrayParaTensores(in);
		Tensor[] y = jnn.arrayParaTensores(out);
		DataLoader dl = new DataLoader(x, y);
		dl.transformY(t -> t.div(255));// normalizar saída de 0-255 para 0-1
		dl.print();

		Modelo modelo = criarSequencial(tamEntrada, tamSaida);
		modelo.print();

		System.out.println("Treinando.");
		long tempo = treinoEmPainel(modelo, dl, imagem.getWidth(), imagem.getHeight());
		long hrs = TimeUnit.NANOSECONDS.toHours(tempo);
		long min = TimeUnit.NANOSECONDS.toMinutes(tempo);
		long sec = TimeUnit.NANOSECONDS.toSeconds(tempo);

		System.out.println("Perda = " + modelo.avaliar(x, y).item());
		System.out.println("Tempo de treinamento: " + hrs + "h " + min + "m " + sec + "s");

		if (historico) {
			exportarHistorico(modelo, CAMINHO_HISTORICO);
			executarComando("python grafico.py " + CAMINHO_HISTORICO);
		}
	}

	/**
	 * Cria um modelo MLP.
	 * @param in quantidade de dados de entrada.
	 * @param out quantidade de dados de saída.
	 * @return {@code Modelo} criado.
	 */
	static Modelo criarRna(int in, int out) {
		RedeNeural modelo = new RedeNeural(in, 8, 8, out);
		
		Object otm = new SGD(0.001, 0.99);
		Object loss = "mse";
		modelo.compilar(otm, loss);

		modelo.configurarAtivacao("sigmoid");
		modelo.configurarAtivacao(modelo.camadaSaida(), "sigmoid");
		modelo.setHistorico(historico);
		
		return modelo;
	}

	/**
	 * Cria um modelo Sequencial.
	 * @param in quantidade de dados de entrada.
	 * @param out quantidade de dados de saída.
	 * @return {@code Modelo} criado.
	 */
	static Modelo criarSequencial(int in, int out) {
		Sequencial modelo = new Sequencial(
			new Entrada(in),
			new Densa(20, "tanh"),
			new Densa(20, "tanh"),
			new Densa(out, "sigmoid")
		);

		Object optm = "sgd";
		Object loss = "mse"; 

		modelo.compilar(optm, loss);
		modelo.setHistorico(historico);

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
		final int EPOCAS_POR_FRAME = 55;

		//acelerar o processo de desenho
		//bom em situações de janelas muito grandes
		int n = Runtime.getRuntime().availableProcessors();
		int numThreads = (n > 1) ? (int)(n * 0.25) : 2;

		JanelaTreino jt = new JanelaTreino(largura, altura, ESCALA_RENDER, numThreads);
		jt.desenharTreino(modelo, 0);
		
		//trabalhar com o tempo de renderização baseado no fps
		double intervaloDesenho = 1_000_000_000/FPS;
		double proximoTempoDesenho = System.nanoTime() + intervaloDesenho;
		double tempoRestante;

		int i = 0;
		long tempoTreino = System.nanoTime();
		while (i < EPOCAS && jt.isVisible()) {
			modelo.treinar(loader, EPOCAS_POR_FRAME, false);
			jt.desenharTreino(modelo, i);
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
		jt.dispose();
		
		return tempoTreino;
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

		try (ExecutorService exec = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors()/2)) {
			final int n = dadosPerdas.length;
			for (int i = 0; i < n; i++) {
				final int id = i;
				exec.submit(() -> {
					dadosPerdas[id][0] = perdas[id];
				});
			}
		} catch (Exception e) {
			throw e;
		}

		Dados dados = new Dados(dadosPerdas);
		ged.exportarCsv(dados, caminho);
	}

	/**
	 * Formata o valor recebido para a quantidade de casas após o ponto
	 * flutuante.
	 * @param x valor alvo.
	 * @param casas quantidade de casas após o ponto flutuante.
	 * @return
	 */
	static String formatarDecimal(double x, int casas) {
		String formato = "#." + "#".repeat(casas);
		return new DecimalFormat(formato).format(x);
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
						in.set(((double)x / (larguraImagem-1)), 0);
						in.set(((double)y / (alturaImagem-1)), 1);
						double[] saida = new double[1];
					
						clones[id].forward(in);
					
						saida[0] = clones[id].saidaParaArray()[0] * 255;

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
						double[] entrada = new double[2];
						in.set(((double)x / (larguraImagem-1)), 0);
						in.set(((double)y / (alturaImagem-1)), 1);
						double[] saida = new double[3];

						entrada[0] = (double)x / (larguraImagem-1);
						entrada[1] = (double)y / (alturaImagem-1);
					
						redes[id].forward(in);
						double[] s = redes[id].saidaParaArray();
					
						saida[0] = s[0] * 255;
						saida[1] = s[1] * 255;
						saida[2] = s[2] * 255;

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
	static double[][] imagemParaDadosTreinoEscalaCinza(BufferedImage imagem) {
		if (imagem == null) throw new IllegalArgumentException("A imagem fornecida é nula.");

		int larguraImagem = imagem.getWidth();
		int alturaImagem = imagem.getHeight();

		double[][] dadosImagem = new double[larguraImagem * alturaImagem][3];
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
				double xNormalizado = (double) x / (larguraImagem - 1);
				double yNormalizado = (double) y / (alturaImagem - 1);
				double escalaCinza = (r + g + b) / 3.0;
				
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
	static double[][] imagemParaDadosTreinoRGB(BufferedImage imagem) {
		if (imagem == null) throw new IllegalArgumentException("A imagem fornecida é nula.");
		int larguraImagem = imagem.getWidth();
		int alturaImagem = imagem.getHeight();

		double[][] dadosImagem = new double[larguraImagem * alturaImagem][5];
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
				double xNormalizado = (double) x / (larguraImagem-1);
				double yNormalizado = (double) y / (alturaImagem-1);  
				double rNormalizado = (double) r / 255;
				double gNormalizado = (double) g / 255;
				double bNormalizado = (double) b / 255;

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