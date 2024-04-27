package testes.modelos;

import java.text.DecimalFormat;

import jnn.camadas.Camada;
import jnn.camadas.Densa;
import jnn.camadas.Dropout;
import jnn.camadas.Entrada;
import jnn.modelos.Modelo;
import jnn.modelos.Sequencial;
import lib.ged.Dados;
import lib.ged.Ged;

public class Classificador{
	static Ged ged = new Ged();
	
	public static void main(String[] args){
		ged.limparConsole();

		//carregando dados e tratando
		//removendo linha com nomes das categorias
		//tranformando a ultima coluna em categorização binária
		Dados iris = ged.lerCsv("./dados/csv/iris.csv");
		ged.remLin(iris, 0);
		int[] shape = ged.shapeDados(iris);
		int ultimoIndice = shape[1]-1;
		ged.categorizar(iris, ultimoIndice);
		System.out.println("Tamanho dados = " + iris.shapeInfo());

		//separando dados de treino e teste
		double[][] dados = ged.dadosParaDouble(iris);
		ged.embaralharDados(dados);
		double[][][] treinoTeste = (double[][][]) ged.separarTreinoTeste(dados, 0.25f);
		double[][] treino = treinoTeste[0];
		double[][] teste = treinoTeste[1];
		int qEntradas = 4;// dados de entrada (features)
		int qSaidas = 3;// classificações (class)

		var treinoX = (double[][]) ged.separarDadosEntrada(treino, qEntradas);
		var treinoY = (double[][]) ged.separarDadosSaida(treino, qSaidas);

		var testeX = (double[][]) ged.separarDadosEntrada(teste, qEntradas);
		var testeY = (double[][]) ged.separarDadosSaida(teste, qSaidas);

		//criando e configurando a rede neural
		Sequencial modelo = new Sequencial(new Camada[]{
			new Entrada(qEntradas),
			new Densa(12, "sigmoid"),
			new Dropout(0.2),
			new Densa(12, "sigmoid"),
			new Dropout(0.2),
			new Densa(qSaidas, "softmax")
		});

		modelo.compilar("sgd", "entropia-cruzada");
		modelo.setHistorico(true);
		// modelo.info();
		
		//treinando e avaliando os resultados
		modelo.treinar(treinoX, treinoY, 180, 16, true);
		double acc = modelo.avaliador().acuracia(testeX, testeY);
		System.out.println("Acurácia = " + formatarDecimal(acc*100, 4) + "%");
		System.out.println("Perda = " + modelo.avaliar(testeX, testeY));

		int[][] matrizConfusao = modelo.avaliador().matrizConfusao(testeX, testeY);
		Dados d = new Dados(matrizConfusao);
		d.editarNome("Matriz de confusão");
		d.print();

		exportarHistorico(modelo, "historico-perda");
		// compararSaidaRede(modelo, testeX, testeY, "");
		executarComando("python grafico.py historico-perda");
	}

	public static void compararSaidaRede(Sequencial rede, double[][] dadosEntrada, double[][] dadosSaida, String texto){
		int nEntrada = rede.camada(0).formatoEntrada()[1];
		int nSaida = rede.camadaSaida().tamanhoSaida();

		double[] entradaRede = new double[nEntrada];
		double[] saidaRede = new double[nSaida];

		System.out.println("\n" + texto);

		//mostrar saída da rede comparada aos dados
		for(int i = 0; i < dadosEntrada.length; i++){
			for(int j = 0; j < dadosEntrada[0].length; j++){
				entradaRede[j] = dadosEntrada[i][j];
			}

			rede.forward(entradaRede);
			saidaRede = rede.saidaParaArray();

			//apenas formatação
			if(i < 10) System.out.print("Dado 00" + i + " |");
			else if(i < 100) System.out.print("Dado 0" + i + " |");
			else System.out.print("Dado " + i + " |");
			for(int j = 0; j < entradaRede.length; j++){
				System.out.print(" " + entradaRede[j] + " ");
			}

			System.out.print(" - ");
			for(int j = 0; j < dadosSaida[0].length; j++){
				System.out.print(" " + dadosSaida[i][j]);
			}
			System.out.print(" | Rede ->");
			for(int j = 0; j < nSaida; j++){
				System.out.print("  " + formatarDecimal(saidaRede[j], 4));
			}
			System.out.println();
		}
	}

	public static String formatarDecimal(double valor, int casas){
		String valorFormatado = "";

		String formato = "#.";
		for(int i = 0; i < casas; i++) formato += "#";

		DecimalFormat df = new DecimalFormat(formato);
		valorFormatado = df.format(valor);

		return valorFormatado;
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

	public static void executarComando(String comando){
		try{
			new ProcessBuilder("cmd", "/c", comando).inheritIO().start().waitFor();
		}catch(Exception e){
			e.printStackTrace();
		}
	}
}
