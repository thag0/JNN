package testes.modelos;

import jnn.Funcional;
import jnn.camadas.Densa;
import jnn.camadas.Entrada;
import jnn.core.tensor.Tensor;
import jnn.modelos.*;
import lib.ged.Dados;
import lib.ged.Ged;

public class TesteModelos{
	static Ged ged = new Ged();
	static Funcional jnn = new Funcional();

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

		Tensor[] treinoX = jnn.arrayParaTensores(entrada);
		Tensor[] treinoY = jnn.arrayParaTensores(saida);

		int nEntradas = entrada[0].length;
		int nSaidas = saida[0].length;
		int nOcultas = 3;
		long seed = 12345678L;
		int epocas = 15_000;

		String atv1 = "sigmoid";
		String atv2 = "sigmoid";
		String otm = "sgd";
		String perda = "mse";

		Sequencial seq = new Sequencial(
			new Entrada(nEntradas),
			new Densa(nOcultas, atv1),
			new Densa(nOcultas, atv1),
			new Densa(nSaidas, atv2)
		);
		seq.setSeed(seed);
		seq.compilar(otm, perda);
		
		RedeNeural rna = new RedeNeural(new int[]{nEntradas, nOcultas, nOcultas, nSaidas});
		rna.setSeed(seed);
		rna.compilar(otm, perda);
		rna.configurarAtivacao(atv1);
		rna.configurarAtivacao(rna.camadaSaida(), atv2);
		    
		rna.treinar(treinoX, treinoY, epocas, false);
		seq.treinar(treinoX, treinoY, epocas, false);

		double perdaSeq = seq.avaliador().erroMedioQuadrado(treinoX, treinoY).item();
		double perdaRna = rna.avaliador().erroMedioQuadrado(treinoX, treinoY).item();

		System.out.println("Perda Seq: " + perdaSeq);
		System.out.println("Perda Rna: " + perdaRna);
		System.out.println("Diferença RNA - SEQ = " + (perdaRna - perdaSeq));//esperado 0
		
		System.out.println();
		for(int i = 0; i < 2; i++){
			for(int j = 0; j < 2; j++){
				Tensor e = new Tensor(new double[] {i, j}, 2);
				double prevSeq = seq.forward(e).get(0);
				double prevRna = rna.forward(e).get(0);
				System.out.println(i + " " + j + " - Rna: " + prevRna + "      \t    Seq: " + prevSeq);
			}
		}
	}

	/**
	 * Salva um arquivo csv com o historico de desempenho da rede.
	 * @param modelo modelo de rede neural.
	 * @param ged gerenciador de dados.
	 */
	public static void exportarHistoricoPerda(Sequencial modelo){
		System.out.println("Exportando histórico de perda");
		double[] perdas = modelo.hist();
		double[][] dadosPerdas = new double[perdas.length][1];

		for(int i = 0; i < dadosPerdas.length; i++){
			dadosPerdas[i][0] = perdas[i];
		}

		Dados dados = new Dados(dadosPerdas);
		ged.exportarCsv(dados, "historico-perda");
	}
}