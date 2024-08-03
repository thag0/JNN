package testes.modelos;

import java.text.DecimalFormat;

import jnn.camadas.Densa;
import jnn.camadas.Dropout;
import jnn.camadas.Entrada;
import jnn.core.Utils;
import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;
import jnn.modelos.Sequencial;
import lib.ged.Dados;
import lib.ged.Ged;

public class Classificador{
	static Ged ged = new Ged();
	static Utils utils = new Utils();
	
	public static void main(String[] args){
		ged.limparConsole();

		//carregando dados e tratando
		//removendo linha com nomes das categorias
		//tranformando a ultima coluna em categorização binária
		Dados iris = ged.lerCsv("./dados/csv/iris.csv");
		ged.remLin(iris, 0);
		int[] shape = iris.shape();
		int ultimoIndice = shape[1]-1;
		ged.categorizar(iris, ultimoIndice);
		System.out.println("Tamanho dados = " + iris.shapeStr());

		// separando dados de treino e teste
		double[][] dados = ged.dadosParaDouble(iris);
		ged.embaralharDados(dados);
		double[][][] treinoTeste = (double[][][]) ged.separarTreinoTeste(dados, 0.25f);
		double[][] treino = treinoTeste[0];
		double[][] teste = treinoTeste[1];
		int numEntradas = 4;// dados de entrada (features)
		int numSaidas = 3;// classificações (class)

		// carregando dados de treino e teste
		double[][] inTreino  = (double[][]) ged.separarDadosEntrada(treino, numEntradas);
		double[][] outTreino = (double[][]) ged.separarDadosSaida(treino, numSaidas);
		Tensor[] treinoX = utils.arrayParaTensores(inTreino);
		Tensor[] treinoY = utils.arrayParaTensores(outTreino);
		
		double[][] inTeste  = (double[][]) ged.separarDadosEntrada(teste, numEntradas);
		double[][] outTeste = (double[][]) ged.separarDadosSaida(teste, numSaidas);
		Tensor[] testeX = utils.arrayParaTensores(inTeste);
		Tensor[] testeY = utils.arrayParaTensores(outTeste);

		// criando e configurando o modelo
		Sequencial modelo = new Sequencial(
			new Entrada(numEntradas),
			new Densa(12, "sigmoid"),
			new Dropout(0.3),
			new Densa(12, "sigmoid"),
			new Dropout(0.3),
			new Densa(numSaidas, "softmax")
		);

		modelo.compilar("lion", "entropia-cruzada");
		modelo.setHistorico(true);
		modelo.print();
		
		//treinando e avaliando os resultados
		modelo.treinar(treinoX, treinoY, 180, 6, true);
		double acc = modelo.avaliador().acuracia(testeX, testeY).item();
		System.out.println("Acurácia = " + formatarDecimal(acc*100, 4) + "%");
		System.out.println("Perda = " + modelo.avaliar(testeX, testeY).item());

		Tensor matriz = modelo.avaliador().matrizConfusao(testeX, testeY);
		matriz.nome("Matriz de confusão");
		matriz.print();

		exportarHistorico(modelo, "historico-perda");
		// compararSaidaRede(modelo, testeX, testeY, "");
		executarComando("python grafico.py historico-perda");
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
		double[] perdas = modelo.hist();
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
