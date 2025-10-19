import java.awt.image.BufferedImage;
import java.text.DecimalFormat;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import jnn.Funcional;
import jnn.camadas.Densa;
import jnn.camadas.Entrada;
import jnn.core.tensor.Tensor;
import jnn.dataloader.DataLoader;
import jnn.modelos.Modelo;
import jnn.modelos.RedeNeural;
import jnn.modelos.Sequencial;
import jnn.otimizadores.*;
import lib.ged.*;
import lib.geim.Geim;
import render.JanelaTreino;

public class MainImg {
	static Ged ged = new Ged();
	static Geim geim = new Geim();
	static Funcional jnn = new Funcional();

	static final int EPOCAS = 5 * 1000;
	static final double ESCALA_RENDER = 10;
	static final boolean historico = true;
	static final String CAMINHO_HISTORICO = "historico-perda";
	static final String CAMINHO_IMAGGEM = "./dados/mnist/treino/8/img_0.jpg";
	// static final String caminhoImagem = "./dados/mnist/treino/7/img_1.jpg";
	// static final String caminhoImagem = "./dados/32x32/circulos.png";

	public static void main(String[] args) {      
		ged.limparConsole();

		BufferedImage imagem = geim.lerImagem(CAMINHO_IMAGGEM);
		int tamEntrada = 2;
		int tamSaida = 1;
		
		double[][] dados;
		if 		(tamSaida == 1) dados = geim.imagemParaDadosTreinoEscalaCinza(imagem);
		else if (tamSaida == 3) dados = geim.imagemParaDadosTreinoRGB(imagem);
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

		double precisao = (1 - modelo.avaliador().erroMedioQuadrado(x, y).item())*100;
		System.out.println("Precisão = " + formatarDecimal(precisao, 2) + "%");
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
}
