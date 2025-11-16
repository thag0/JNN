package exemplos.modelos;

import java.awt.image.BufferedImage;

import ged.Ged;
import geim.imagem.Imagem;
import ged.Dados;
import geim.Geim;
import jnn.Funcional;
import jnn.camadas.*;
import jnn.core.tensor.Tensor;
import jnn.dataloader.DataLoader;
import jnn.modelos.Modelo;
import jnn.modelos.Sequencial;
import jnn.otimizadores.SGD;

/**
 * Exemplo usando um modelo para se adaptar a uma imagem com o objetivo
 * de usá-lo para fazer upscaling.
 * @see {@code origem} {@link https://www.youtube.com/watch?v=ZjxPPvqNp3k&t=3871s}
 */
public class UpscaleImg {
	static Ged ged = new Ged();
	static Geim geim = new Geim();
	static Funcional jnn = new Funcional();

	public static void main(String[] args) {

		ged.limparConsole();

		// Carregando imagem
		String caminho = "./dados/mnist/treino/8/img_0.jpg";
		BufferedImage imagem = geim.lerImagem(caminho);

		// Tratando dados
		double[][] dados = imagemParaDadosTreinoEscalaCinza(imagem);
		int nEntrada = 2;// posição x y do pixel
		int nSaida = 1;// valor de escala de cinza/brilho do pixel
		DataLoader img = jnn.dataloader(dados, nEntrada, nSaida);
		img.transformY(a -> a.div(255));// normalizar entre 0 e 1

		// Criando modelo
		// -Neste exemplo queremos que o modelo tenha overfitting
		Sequencial modelo = new Sequencial(
			new Entrada(nEntrada),
			new Densa(12, "sigmoid"),
			new Densa(12, "sigmoid"),
			new Densa(nSaida, "sigmoid")
		);

		modelo.setHistorico(true);
		modelo.compilar(new SGD(0.0001, 0.999), "mse");
		modelo.treinar(img, 5_000, true);

		// Avaliando o modelo
		Tensor[] xs = img.getX();
		Tensor[] ys = img.getY();
		double precisao = 1 - modelo.avaliador().erroMedioAbsoluto(xs, ys).item();
		System.out.println("Precisão = " + (precisao * 100));
		System.out.println("Perda = " + modelo.avaliar(xs, ys).item());

		// Salvando dados
		exportarHistoricoPerda(modelo.hist());// histórico de treino para plot
		exportarImagemEscalaCinza(modelo, imagem.getHeight(), imagem.getWidth(), 50, "img.png");// imagem ampliada
	}

	/**
	 * Salva um arquivo csv com o historico de desempenho da rede.
	 * @param modelo modelo.
	 */
	public static void exportarHistoricoPerda(double[] hist) {
		System.out.println("Exportando histórico de perda");
		
		double[][] perdas = new double[hist.length][1];
		for (int i = 0; i < perdas.length; i++){
			perdas[i][0] = hist[i];
		}

		Dados dados = new Dados(perdas);
		ged.exportarCsv(dados, "historico-perda");
	}

	/**
	 * Gera um conjunto de dados de treino baseado na imagem.
	 * @param img imagem base.
	 * @return dados de treino.
	 */
	static double[][] imagemParaDadosTreinoEscalaCinza(BufferedImage img) {
		if (img == null) throw new IllegalArgumentException("A imagem fornecida é nula.");

		int larguraImagem = img.getWidth();
		int alturaImagem = img.getHeight();

		double[][] dadosImagem = new double[larguraImagem * alturaImagem][3];
		int[][] vermelho = geim.getR(img);
		int[][] verde = geim.getG(img);
		int[][] azul = geim.getB(img);

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
		Imagem ampliada = geim.gerarEstruturaImagem(larguraFinal, alturaFinal);

		int alturaImagem = ampliada.altura();
		int larguraImagem = ampliada.largura();

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

						synchronized(ampliada) {
							ampliada.set(x, y, (int)saida[0], (int)saida[0], (int)saida[0]);
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

		ampliada.paraPNG(caminho);
	}

}
