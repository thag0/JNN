package exemplos.modelos;

import java.awt.image.BufferedImage;

import jnn.Funcional;
import jnn.camadas.*;
import jnn.core.tensor.Tensor;
import jnn.dataloader.DataLoader;
import jnn.modelos.Sequencial;
import jnn.otimizadores.SGD;
import lib.ged.Dados;
import lib.ged.Ged;
import lib.geim.Geim;

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
		double[][] dados = geim.imagemParaDadosTreinoEscalaCinza(imagem);
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
		geim.exportarImagemEscalaCinza(imagem, modelo, 50, "img");// imagem ampliada
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
}
