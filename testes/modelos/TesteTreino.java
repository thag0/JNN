package testes.modelos;

import jnn.Funcional;
import jnn.camadas.*;
import jnn.core.tensor.Tensor;
import jnn.modelos.*;
import jnn.otimizadores.SGD;
import lib.ged.Dados;
import lib.ged.Ged;

public class TesteTreino{
	static Ged ged = new Ged();
	static Funcional jnn = new Funcional();

	public static void main(String[] args){
		ged.limparConsole();

		Dados xor = ged.lerCsv("./dados/csv/xor.csv");
		double[][] dados = ged.dadosParaDouble(xor);
		double[][] x = (double[][]) ged.separarDadosEntrada(dados, 2);
		double[][] y = (double[][]) ged.separarDadosSaida(dados, 1);
		int in  = x[0].length;// colunas de entrada
		int out = y[0].length;// colunas de saída

		Tensor[] treinoX = jnn.arrayParaTensores(x);
		Tensor[] treinoY = jnn.arrayParaTensores(y);

		Sequencial modelo = new Sequencial(
			new Entrada(in),
			new Densa(2, "sigmoid"),
			new Densa(out, "sigmoid")
		);
		
		modelo.compilar(new SGD(0.00001, 0.99995), "mse");
		modelo.treinar(treinoX, treinoY, 15_000, false);

		Tensor perda = modelo.avaliar(treinoX, treinoY);
		
		System.out.println("Perda: " + perda.item() + "\n");
		System.out.println(verificar(treinoX, treinoY, modelo));
	}

	static Tensor verificar(Tensor[] entrada, Tensor[] saida, Sequencial modelo){
		Tensor[] preds = modelo.forward(entrada);

		int n = preds.length;
		int cols = 2;
		Tensor res = new Tensor(n, cols);
		for(int i = 0; i < entrada.length; i++){
			res.set(
				saida[i].get(0),
				i, 0
			);
			res.set(
				preds[i].get(0),
				i, 1
			);
		}

		res.nome("Real / Prev");
		return res;
	}

	/**
	 * Salva um arquivo csv com o historico de desempenho do modelo.
	 * @param modelo modelo.
	 * @param caminho caminho onde será salvo o arquivo.
	 */
	static void exportarHistorico(Modelo modelo, String caminho){
		System.out.println("Exportando histórico de perda");
		double[] perdas = modelo.hist();
		double[][] dadosPerdas = new double[perdas.length][1];

		for(int i = 0; i < dadosPerdas.length; i++){
			dadosPerdas[i][0] = perdas[i];
		}

		Dados dados = new Dados(dadosPerdas);
		ged.exportarCsv(dados, caminho);
	}
}
