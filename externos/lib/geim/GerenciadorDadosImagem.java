package externos.lib.geim;

import java.awt.image.BufferedImage;

/**
 * Gerenciador de dados de imagem para o Geim
 */
class GerenciadorDadosImagem {
	
	/**
	 * Contém implementações de manuseio de dados de imagem.
	 */
	public GerenciadorDadosImagem() {}

	/**
	 * 
	 * @param largura
	 * @param altura
	 * @return
	 */
	public Pixel[][] gerarEstruturaImagem(int largura, int altura) {
		if (largura <= 0 || altura <= 0) {
			throw new IllegalArgumentException(
				"Os valores de altura e largura fornecidos são inválidos"
			);
		}

		Pixel[][] dadosImagem = new Pixel[altura][largura];
		for (int y = 0; y < altura; y++) {
			for (int x = 0; x < largura; x++) {
				dadosImagem[y][x] = new Pixel(0, 0, 0);
			}
		}

		return dadosImagem;
	}

	/**
	 * 
	 * @param imagem
	 * @return
	 */
	public Pixel[][] gerarEstruturaImagem(BufferedImage imagem) {
		if (imagem == null) throw new IllegalArgumentException("A imagem fornecida é nula.");
		if (imagem.getWidth() <= 0) throw new IllegalArgumentException("A largura da imagem não pode ser menor ou igual a zero.");
		if (imagem.getHeight() <= 0) throw new IllegalArgumentException("A altura da imagem não pode ser menor ou igual a zero.");
		
		//dimensões da imagem
		int largura = imagem.getWidth();
		int altura = imagem.getHeight();
		Pixel[][] dadosImagem = new Pixel[altura][largura];
		for (int y = 0; y < altura; y++) {         
			for (int x = 0; x < largura; x++) {
				dadosImagem[y][x] = new Pixel();
				dadosImagem[y][x].setR(this.getR(imagem, x, y));
				dadosImagem[y][x].setG(this.getG(imagem, x, y));
				dadosImagem[y][x].setB(this.getB(imagem, x, y));
			}
		}

		return dadosImagem;
	}

	/**
	 * 
	 * @param estruturaImagem
	 * @param x
	 * @param y
	 * @param r
	 * @param g
	 * @param b
	 */
	public void setCor(Pixel[][] estruturaImagem, int x, int y, int r, int g, int b){
		if (estruturaImagem == null) throw new IllegalArgumentException("A estrutura da imagem é nula");   
		if (y < 0 || y > estruturaImagem.length) {
			throw new IllegalArgumentException("O valor x de fornecido está fora de alcance.");
		}
		if (x < 0 || x > estruturaImagem[0].length) {
			throw new IllegalArgumentException("O valor y de fornecido está fora de alcance.");
		}

		estruturaImagem[y][x].setRGB(r, g, b);
	}

	public void preencher(Pixel[][] estrutura, int r, int g, int b) {
		for (int i = 0; i < estrutura.length; i++) {
			for (int j = 0; j < estrutura[i].length; j++) {
				estrutura[i][j].setRGB(r, g, b);
			}
		}
	}

	public int[][] getVermelho(BufferedImage imagem) {
		int largura = imagem.getWidth();
		int altura = imagem.getHeight();

		int[][] vermelho = new int[altura][largura];
		for(int y = 0; y < altura; y++){
			for(int x = 0; x < largura; x++){
				vermelho[y][x] = getR(imagem, x, y);
			}
		}

		return vermelho;
	}

	/**
	 * 
	 * @param imagem
	 * @return
	 */
	public int[][] getVerde(BufferedImage imagem) {
		int largura = imagem.getWidth();
		int altura = imagem.getHeight();

		int[][] verde = new int[altura][largura];
		for (int y = 0; y < altura; y++) {
			for (int x = 0; x < largura; x++) {
				verde[y][x] = getG(imagem, x, y);
			}
		}

		return verde;
	}

	/**
	 * 
	 * @param imagem
	 * @return
	 */
	public int[][] getAzul(BufferedImage imagem) {
		int largura = imagem.getWidth();
		int altura = imagem.getHeight();

		int[][] azul = new int[altura][largura];
		for (int y = 0; y < altura; y++) {
			for (int x = 0; x < largura; x++) {
				azul[y][x] = getB(imagem, x, y);
			}
		}

		return azul;
	}

	/**
	 * 
	 * @param imagem
	 * @return
	 */
	public int[][] getCinza(BufferedImage imagem) {
		int largura = imagem.getWidth();
		int altura = imagem.getHeight();

		int[][] cinza = new int[altura][largura];
		for (int y = 0; y < altura; y++) {
			for (int x = 0; x < largura; x++) {
				int r = getR(imagem, x, y);
				int g = getG(imagem, x, y);
				int b = getB(imagem, x, y);
				int c = (r + g + b) / 3;
				cinza[y][x] = c;
			}
		}

		return cinza;
	}

	/**
	 * 
	 * @param imagem
	 * @param x
	 * @param y
	 * @return
	 */
	public int getR(BufferedImage imagem, int x, int y) {
		int rgb = imagem.getRGB(x, y);
		int r = (rgb >> 16) & 0xFF;
		return r;
	}

	/**
	 * 
	 * @param imagem
	 * @param x
	 * @param y
	 * @return
	 */
	public int getG(BufferedImage imagem, int x, int y) {
		int rgb = imagem.getRGB(x, y);
		int g = (rgb >> 8) & 0xFF;
		return g;
	}

	/**
	 * 
	 * @param imagem
	 * @param x
	 * @param y
	 * @return
	 */
	public int getB(BufferedImage imagem, int x, int y) {
		int rgb = imagem.getRGB(x, y);
		int b = rgb & 0xFF;
		return b;
	}

	/**
	 * 
	 * @param imagem
	 * @return
	 */
	public int[][] getDadosImagem(BufferedImage imagem) {
		int largura = imagem.getWidth();
		int altura = imagem.getHeight();
		int informacoes = 1 + 1 + 3;// x + y + r + g + b
		
		int[][] dados = new int[largura*altura][informacoes];
		
		for (int y = 0; y < largura; y++) {
			for (int x = 0; x < altura; x++) {
				int r = getR(imagem, x, y);
				int g = getG(imagem, x, y);
				int b = getB(imagem, x, y);
			
				int linha = y * largura + x;
				dados[linha][0] = x;
				dados[linha][1] = y;
				dados[linha][2] = r;
				dados[linha][3] = g;
				dados[linha][4] = b;
			}
		}
		
		return dados;
	}

	/**
	 * 
	 * @param imagem
	 * @return
	 */
	public double[][] imagemParaDadosTreinoEscalaCinza(BufferedImage imagem) {
		if (imagem == null) throw new IllegalArgumentException("A imagem fornecida é nula.");

		int larguraImagem = imagem.getWidth();
		int alturaImagem = imagem.getHeight();

		double[][] dadosImagem = new double[larguraImagem * alturaImagem][3];

		int contador = 0;
		for (int y = 0; y < alturaImagem; y++) {
			for (int x = 0; x < larguraImagem; x++) {
				int r = this.getR(imagem, x, y);
				int g = this.getG(imagem, x, y);
				int b = this.getB(imagem, x, y);

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
	 * 
	 * @param imagem
	 * @return
	 */
	public double[][] imagemParaDadosTreinoRGB(BufferedImage imagem) {
		if (imagem == null) throw new IllegalArgumentException("A imagem fornecida é nula.");
		int larguraImagem = imagem.getWidth();
		int alturaImagem = imagem.getHeight();

		double[][] dadosImagem = new double[larguraImagem * alturaImagem][5];

		int contador = 0;

		for (int y = 0; y < alturaImagem; y++) {
			for (int x = 0; x < larguraImagem; x++) {
				int r = this.getR(imagem, x, y);
				int g = this.getG(imagem, x, y);
				int b = this.getB(imagem, x, y);

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
