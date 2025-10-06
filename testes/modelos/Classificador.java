package testes.modelos;

import java.text.DecimalFormat;

import jnn.Funcional;

import jnn.camadas.Densa;
import jnn.camadas.Dropout;
import jnn.camadas.Entrada;

import jnn.core.Utils;
import jnn.core.tensor.Tensor;

import jnn.dataloader.DataLoader;

import jnn.modelos.Sequencial;

import lib.ged.Dados;
import lib.ged.Ged;

/**
 * Exemplo de criação, treino e validação de um modelo da biblioteca
 * fazendo uso do dataset {@code Iris}
 * @see {@code Iris:} {@link https://gist.github.com/netj/8836201}
 */
public class Classificador {
	static Ged ged = new Ged();
	static Utils utils = new Utils();
	static Funcional jnn = new Funcional();
	
	public static void main(String[] args){
		ged.limparConsole();

		// Carregando dados e pré-processando
		Dados iris = ged.lerCsv("./dados/csv/iris.csv");
		ged.remLin(iris, 0);// Removendo linha com nomes das categorias
		int[] shape = iris.shape();
		int ultimoIndice = shape[1]-1;
		ged.categorizar(iris, ultimoIndice);// Tranformando a ultima coluna em categorização binária

		// Separando os dados em treino e teste.
		int numEntradas = 4;// dados de entrada (features)
		int numSaidas = 3;// rótulos (classes)
		double[][] dados = ged.dadosParaDouble(iris);
		DataLoader dIris = jnn.dataloader(dados, numEntradas, numSaidas);// Gerando o dataset
		dIris.embaralhar();
		DataLoader[] ds = dIris.separar(0.75, 0.25);// separando 75% treino, 25% teste
		DataLoader treino = ds[0];
		DataLoader teste  = ds[1];
		
		// Criando um modelo
		Sequencial modelo = new Sequencial(
			new Entrada(numEntradas),
			new Densa(12, "tanh"),
			new Dropout(0.25),
			new Densa(12, "tanh"),
			new Dropout(0.25),
			new Densa(numSaidas, "softmax")
		);
			
		modelo.compilar("adam", "entropia-cruzada");
		modelo.setHistorico(true);
		
		treino.print();
		modelo.print();

		// Treinando
		modelo.treinar(treino, 500, 12, true);
		
		// Avaliando o modelo
		Tensor[] testeX = teste.getX();
		Tensor[] testeY = teste.getY();
		double acc = modelo.avaliador().acuracia(testeX, testeY).item();
		System.out.println("Acurácia = " + formatarDecimal(acc*100, 4) + "%");
		System.out.println("Perda = " + modelo.avaliar(testeX, testeY).item());

		// Matriz de confusão
		Tensor matriz = modelo.avaliador().matrizConfusao(testeX, testeY);
		matriz.nome("Matriz de confusão").print();

		exportarHistorico(modelo.hist(), "historico-perda");
		executarComando("python grafico.py historico-perda");
	}

	/**
	 * Formata o valor em uma String.
	 * @param valor valor base.
	 * @param casas casas decimais.
	 * @return {@code String} com valor formatado
	 */
	public static String formatarDecimal(double valor, int casas) {
		String formato = "#." + "#".repeat(casas);
		String valStr = new DecimalFormat(formato).format(valor);
		return valStr;
	}

	/**
	 * Salva um arquivo csv com o historico de desempenho do modelo.
	 * @param modelo modelo.
	 * @param caminho caminho onde será salvo o arquivo.
	 */
	static void exportarHistorico(double[] hist, String caminho){
		System.out.println("Exportando histórico de perda");
		double[][] dadosPerdas = new double[hist.length][1];

		for(int i = 0; i < dadosPerdas.length; i++){
			dadosPerdas[i][0] = hist[i];
		}

		Dados dados = new Dados(dadosPerdas);
		ged.exportarCsv(dados, caminho);
	}

	/**
	 * Experimental
	 * @param comando comando para executar no prompt.
	 */
	static void executarComando(String comando){
		try{
			new ProcessBuilder("cmd", "/c", comando).inheritIO().start().waitFor();
		}catch(Exception e){
			e.printStackTrace();
		}
	}
}
