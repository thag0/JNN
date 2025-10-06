package testes.modelos;

import java.awt.image.BufferedImage;

import jnn.Funcional;
import jnn.camadas.*;
import jnn.dataloader.DataLoader;
import jnn.modelos.Modelo;
import jnn.modelos.Sequencial;
import jnn.otimizadores.SGD;
import lib.ged.Dados;
import lib.ged.Ged;
import lib.geim.Geim;

public class AmpliarImagem{
	public static void main(String[] args){
		Ged ged = new Ged();
		Geim geim = new Geim();
		Funcional jnn = new Funcional();

		ged.limparConsole();

		//importando imagem para treino da rede
		final String caminho = "./dados/mnist/treino/8/img_0.jpg";
		BufferedImage imagem = geim.lerImagem(caminho);
		double[][] dados = geim.imagemParaDadosTreinoEscalaCinza(imagem);
		int nEntrada = 2;// posição x y do pixel
		int nSaida = 1;// valor de escala de cinza/brilho do pixel

		//preparando dados para treinar a rede
		double[][] in = (double[][]) ged.separarDadosEntrada(dados, nEntrada);
		double[][] out = (double[][]) ged.separarDadosSaida(dados, nSaida);
		
		DataLoader dataloader = new DataLoader(
			jnn.arrayParaTensores(in),
			jnn.arrayParaTensores(out)
		);

		dataloader.transformY(a -> a.div(255));// normalizar entre 0 e 1

		//criando rede neural para lidar com a imagem
		//nesse exemplo queremos que ela tenha overfitting
		Sequencial modelo = new Sequencial(
			new Entrada(nEntrada),
			new Densa(14, "tanh"),
			new Densa(14, "tanh"),
			new Densa(nSaida, "sigmoid")
		);
		modelo.setHistorico(true);
		modelo.compilar(new SGD(0.0001, 0.999), "mse");
		modelo.treinar(dataloader, 3_000, true);

		//avaliando resultados
		double precisao = 1 - modelo.avaliador().erroMedioAbsoluto(dataloader.getX(), dataloader.getY()).item();
		System.out.println("Precisão = " + (precisao * 100));
		System.out.println("Perda = " + modelo.avaliar(dataloader.getX(), dataloader.getY()).item());

		exportarHistoricoPerda(modelo, ged);
		geim.exportarImagemEscalaCinza(imagem, modelo, 50, "img");
	}

	/**
	 * Salva um arquivo csv com o historico de desempenho da rede.
	 * @param rede rede neural.
	 * @param ged gerenciador de dados.
	 */
	public static void exportarHistoricoPerda(Modelo rede, Ged ged){
		System.out.println("Exportando histórico de perda");
		double[] perdas = rede.hist();
		double[][] dadosPerdas = new double[perdas.length][1];

		for(int i = 0; i < dadosPerdas.length; i++){
			dadosPerdas[i][0] = perdas[i];
		}

		Dados dados = new Dados(dadosPerdas);
		ged.exportarCsv(dados, "historico-perda");
	}
}
