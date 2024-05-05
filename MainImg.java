import java.awt.image.BufferedImage;
import java.text.DecimalFormat;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import jnn.camadas.*;
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
	static final int EPOCAS = 8*1000;
	static final double ESCALA_RENDER = 9;
	static boolean calcularHistorico = true;
	static final String CAMINHO_HISTORICO = "historico-perda";
	static final String CAMINHO_IMAGGEM = "./dados/mnist/treino/8/img_0.jpg";
	// static final String caminhoImagem = "./dados/mnist/treino/7/img_1.jpg";
	// static final String caminhoImagem = "./dados/32x32/circulos.png";

	public static void main(String[] args) {      
		ged.limparConsole();

		int tamEntrada = 2;
		int tamSaida = 1;
		BufferedImage imagem = geim.lerImagem(CAMINHO_IMAGGEM);
		
		double[][] dados;
		if (tamSaida == 1) dados = geim.imagemParaDadosTreinoEscalaCinza(imagem);
		else if (tamSaida == 3) dados = geim.imagemParaDadosTreinoRGB(imagem);
		else return;

		double[][] in  = (double[][]) ged.separarDadosEntrada(dados, tamEntrada);
		double[][] out = (double[][]) ged.separarDadosSaida(dados, tamSaida);

		Modelo modelo = criarSequencial(tamEntrada, tamSaida);
		modelo.info();

		//treinar e marcar tempo
		long horas, minutos, segundos;

		System.out.println("Treinando.");
		long tempoDecorrido = treinoEmPainel(modelo, imagem.getWidth(), imagem.getHeight(), in, out);

		long segundosTotais = TimeUnit.NANOSECONDS.toSeconds(tempoDecorrido);
		horas = segundosTotais / 3600;
		minutos = (segundosTotais % 3600) / 60;
		segundos = segundosTotais % 60;

		double precisao = (1 - modelo.avaliador().erroMedioQuadrado(in, out))*100;
		System.out.println("Precisão = " + formatarDecimal(precisao, 2) + "%");
		System.out.println("Perda = " + modelo.avaliar(in, out));
		System.out.println("Tempo de treinamento: " + horas + "h " + minutos + "m " + segundos + "s");

		if (calcularHistorico) {
			exportarHistorico(modelo, CAMINHO_HISTORICO);
			executarComando("python grafico.py " + CAMINHO_HISTORICO);
		}
	}

	static Modelo criarRna(int entradas, int saidas) {
		Otimizador otm = new SGD(0.001, 0.99);

		RedeNeural modelo = new RedeNeural(entradas, 8, 8, saidas);
		modelo.compilar(otm, "mse");
		modelo.configurarAtivacao("sigmoid");
		modelo.configurarAtivacao(modelo.camadaSaida(), "sigmoid");
		modelo.setHistorico(calcularHistorico);
		
		return modelo;
	}

	static Modelo criarSequencial(int entradas, int saidas) {
		Sequencial modelo = new Sequencial(new Camada[]{
			new Entrada(entradas),
			new Densa(8, "sigmoid"),
			new Densa(8, "sigmoid"),
			new Densa(saidas, "sigmoid")
		});

		modelo.compilar(new SGD(0.001, 0.99), "mse");
		modelo.setHistorico(calcularHistorico);

		return modelo;
	}

	/**
	 * Treina e exibe o resultado da Rede Neural no painel.
	 * @param modelo modelo de rede neural usado no treino.
	 * @param altura altura da janela renderizada.
	 * @param largura largura da janela renderizada.
	 * @param entradas dados de entrada para o treino.
	 * @param saidas dados de saída relativos a entrada.
	 * @return tempo (em nano segundos) do treino.
	 */
	static long treinoEmPainel(Modelo modelo, int altura, int largura, double[][] entradas, double[][] saidas) {
		final int fps = 600000;
		int epocasPorFrame = 50;

		//acelerar o processo de desenho
		//bom em situações de janelas muito grandes
		int n = Runtime.getRuntime().availableProcessors();
		int numThreads = (n > 1) ? (int)(n * 0.25) : 2;

		JanelaTreino jt = new JanelaTreino(largura, altura, ESCALA_RENDER, numThreads);
		jt.desenharTreino(modelo, 0);
		
		//trabalhar com o tempo de renderização baseado no fps
		double intervaloDesenho = 1_000_000_000/fps;
		double proximoTempoDesenho = System.nanoTime() + intervaloDesenho;
		double tempoRestante;

		int i = 0;
		long tempoTreino = System.nanoTime();
		while (i < EPOCAS && jt.isVisible()) {
			modelo.treinar(entradas, saidas, epocasPorFrame, false);
			jt.desenharTreino(modelo, i);
			i += epocasPorFrame;

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
		double[] perdas = modelo.historico();
		double[][] dadosPerdas = new double[perdas.length][1];

		try (ExecutorService exec = Executors.newFixedThreadPool(2)) {
			exec.execute(() -> {
				for(int i = 0; i < dadosPerdas.length/2; i++){
					dadosPerdas[i][0] = perdas[i];
				}
			});
			exec.execute(() -> {
				for(int i = dadosPerdas.length/2; i < dadosPerdas.length; i++){
					dadosPerdas[i][0] = perdas[i];
				}
			});
		} catch (Exception e) {
			throw e;
		}

		Dados dados = new Dados(dadosPerdas);
		ged.exportarCsv(dados, caminho);
	}

	/**
	 * Formata o valor recebido para a quantidade de casas após o ponto
	 * flutuante.
	 * @param valor valor alvo.
	 * @param casas quantidade de casas após o ponto flutuante.
	 * @return
	 */
	static String formatarDecimal(double valor, int casas) {
		String valorFormatado = "";

		String formato = "#.";
		for (int i = 0; i < casas; i++) formato += "#";

		DecimalFormat df = new DecimalFormat(formato);
		valorFormatado = df.format(valor);

		return valorFormatado;
	}

	/**
	 * teste
	 * @param comando
	 */
	public static void executarComando(String comando){
		try {
			new ProcessBuilder("cmd", "/c", comando).inheritIO().start().waitFor();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
