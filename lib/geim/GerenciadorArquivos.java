package lib.geim;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.FileInputStream;

import javax.imageio.ImageIO;

import jnn.core.tensor.Variavel;
import jnn.modelos.Modelo;

/**
 * Gerenciador de arquivos do Geim
 */
class GerenciadorArquivos {
	
	/**
	 * Contém as implementações de gestão de arquivos do Geim
	 */
	public GerenciadorArquivos() {}

	/**
	 * 
	 * @param caminho
	 * @return
	 */
	public BufferedImage lerImagem(String caminho) {
		File arquivo = new File(caminho);
		if (!arquivo.exists()) {
			throw new IllegalArgumentException(
				"Diretório \"" + caminho + "\" não encontrado."
			);
		}
  
		BufferedImage img = null;
		
		try {
			img = ImageIO.read(new FileInputStream(arquivo));
		} catch (IOException e) {
			System.out.println("Erro ao ler a imagem \"" + caminho + "\"");
			e.printStackTrace();
		}
  
		return img;
	}
  
	/**
	 * 
	 * @param estruturaImagem
	 * @param caminho
	 */
	public void exportarPng(Pixel[][] estruturaImagem, String caminho) {
		int alturaImagem = estruturaImagem.length;
		int larguraImagem = estruturaImagem[0].length;

		BufferedImage imagem = new BufferedImage(larguraImagem, alturaImagem, BufferedImage.TYPE_INT_RGB);

		int r = 0;
		int g = 0;
		int b = 0;
		int rgb = 0;

		for (int y = 0; y < alturaImagem; y++) {
			for (int x = 0; x < larguraImagem; x++) {
				r = estruturaImagem[y][x].getR();
				g = estruturaImagem[y][x].getG();
				b = estruturaImagem[y][x].getB();

				rgb = (r << 16) | (g << 8) | b;
				imagem.setRGB(x, y, rgb);
			}
		}

		try {
			File arquivo = new File((caminho + ".png"));
			ImageIO.write(imagem, "png", arquivo);

		} catch (Exception e) {
			System.out.println("Erro ao exportar imagem");
			e.printStackTrace();
		}
	}

	/**
	 * 
	 * @param gdi
	 * @param imagem
	 * @param modelo
	 * @param escala
	 * @param caminho
	 */
	public void exportarImagemEscalaCinza(GerenciadorDadosImagem gdi, BufferedImage imagem, Modelo modelo, double escala, String caminho) {
		if (imagem == null) throw new IllegalArgumentException("A imagem fornecida é nula.");
		if (escala <= 0) throw new IllegalArgumentException("O valor de escala não pode ser menor que 1.");
		if (modelo.camadaSaida().tamSaida() != 1) {
			throw new IllegalArgumentException(
				"O modelo deve trabalhar apenas com uma unidade na camada de saída para a escala de cinza."
			);
		}

		int larguraFinal = (int)(imagem.getWidth() * escala);
		int alturaFinal = (int)(imagem.getHeight() * escala);

		Pixel[][] imagemAmpliada = gdi.gerarEstruturaImagem(larguraFinal, alturaFinal);
		
		int alturaImagem = imagemAmpliada.length;
		int larguraImagem = imagemAmpliada[0].length;

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
				for (int y = inicio; y < fim; y++) {
					for (int x = 0; x < larguraImagem; x++) {
						double[] entrada = new double[2];
						double[] saida = new double[1];

						entrada[0] = (double)x / (larguraImagem-1);
						entrada[1] = (double)y / (alturaImagem-1);
					
						clones[id].forward(entrada);
					
						saida[0] = clones[id].saidaParaArray()[0].get() * 255;

						synchronized(imagemAmpliada) {
							gdi.setCor(imagemAmpliada, x, y, (int)saida[0], (int)saida[0], (int)saida[0]);
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

		this.exportarPng(imagemAmpliada, caminho);
	}

	/**
	 * 
	 * @param gdi
	 * @param imagem
	 * @param modelo
	 * @param escala
	 * @param caminho
	 */
	public void exportarImagemRGB(GerenciadorDadosImagem gdi, BufferedImage imagem, Modelo modelo, double escala, String caminho) {
		if (imagem == null) throw new IllegalArgumentException("A imagem fornecida é nula.");
		if (escala <= 0) throw new IllegalArgumentException("O valor de escala não pode ser menor que 1.");
		if (modelo.camadaSaida().tamSaida() != 3) {
			throw new IllegalArgumentException(
				"O modelo deve trabalhar apenas com três neurônios na saída para RGB."
			);
		}

		//estrutura de dados da imagem
		int larguraFinal = (int)(imagem.getWidth() * escala);
		int alturaFinal = (int)(imagem.getHeight() * escala);
		Pixel[][] imagemAmpliada = gdi.gerarEstruturaImagem(larguraFinal, alturaFinal);
		
		int alturaImagem = imagemAmpliada.length;
		int larguraImagem = imagemAmpliada[0].length;

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
				for (int y = inicio; y < fim; y++) {
					for (int x = 0; x < larguraImagem; x++) {
						double[] entrada = new double[2];
						double[] saida = new double[3];

						entrada[0] = (double)x / (larguraImagem-1);
						entrada[1] = (double)y / (alturaImagem-1);
					
						redes[id].forward(entrada);
						Variavel[] s = redes[id].saidaParaArray();
					
						saida[0] = s[0].get() * 255;
						saida[1] = s[1].get() * 255;
						saida[2] = s[2].get() * 255;

						synchronized (imagemAmpliada) {
							gdi.setCor(imagemAmpliada, x, y, (int)saida[0], (int)saida[1], (int)saida[2]);
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

		this.exportarPng(imagemAmpliada, caminho);
	}
}
