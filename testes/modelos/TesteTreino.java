package testes.modelos;

import jnn.camadas.*;
import jnn.core.Utils;
import jnn.core.tensor.Tensor;
import jnn.modelos.*;
import jnn.otimizadores.SGD;
import lib.ged.Dados;
import lib.ged.Ged;

public class TesteTreino{
	static Ged ged = new Ged();
	static Utils utils = new Utils();

	public static void main(String[] args){
		ged.limparConsole();

		double[][] entrada = {
			{0, 0},
			{0, 1},
			{1, 0},
			{1, 1}
		};
		double[][] saida = {
			{0},
			{1},
			{1},
			{0}
		};

		Tensor[] treinoX = utils.array2DParaTensors(entrada);
		Tensor[] treinoY = utils.array2DParaTensors(saida);

		Sequencial modelo = new Sequencial(
			new Entrada(treinoX[0].tamanho()),
			new Densa(3, "sigmoid"),
			new Densa(1, "sigmoid")
		);
		
		modelo.compilar(new SGD(0.00001, 0.9999), "mse");
		modelo.treinar(treinoX, treinoY, 20_000, false);
		verificar(treinoX, treinoY, modelo);

		double perda = modelo.avaliar(treinoX, treinoY).item();
		System.out.println("\nPerda: " + perda + "\n");
	}

	static void verificar(Tensor[] entrada, Tensor[] saida, Sequencial modelo){
		Tensor[] preds = modelo.forwards(entrada);
		
		for(int i = 0; i < entrada.length; i++){
			System.out.println(
				"Real: " + saida[i].get(0) + ",   " +  
				"Prev: " + preds[i].get(0)
			);
		}
	}

	/**
	 * Salva um arquivo csv com o historico de desempenho do modelo.
	 * @param modelo modelo.
	 * @param caminho caminho onde será salvo o arquivo.
	 */
	static void exportarHistorico(Modelo modelo, String caminho){
		System.out.println("Exportando histórico de perda");
		double[] perdas = modelo.historico();
		double[][] dadosPerdas = new double[perdas.length][1];

		for(int i = 0; i < dadosPerdas.length; i++){
			dadosPerdas[i][0] = perdas[i];
		}

		Dados dados = new Dados(dadosPerdas);
		ged.exportarCsv(dados, caminho);
	}
}
