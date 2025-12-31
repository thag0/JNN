package exemplos;

import java.text.DecimalFormat;

import ged.Ged;
import ged.Dados;
import jnn.Funcional;
import jnn.camadas.Densa;
import jnn.camadas.Dropout;
import jnn.camadas.Entrada;

import jnn.dataloader.DataLoader;

import jnn.modelos.Sequencial;

/**
 * Exemplo de criação, treino e validação de um modelo da biblioteca
 * fazendo uso do dataset {@code Iris}
 * @see {@code Iris:} {@link https://gist.github.com/netj/8836201}
 */
public class Iris {

	/**
	 * Gerenciador de dados.
	 */
	static Ged ged = new Ged();
	
	/**
	 * Interface funcional.
	 */
	static Funcional jnn = new Funcional();
	
	static {
		ged.limparConsole();
	}
	
	public static void main(String[] args){
		// Carregando dados e pré-processando
		Dados iris = ged.lerCsv("./dados/csv/iris.csv");
		ged.dropLin(iris, 0);// Removendo linha com nomes das categorias
		int[] shape = iris.shape();
		int ultimoIndice = shape[1]-1;
		ged.categorizar(iris, ultimoIndice);// Tranformando a ultima coluna em categorização binária

		// Separando os dados em treino e teste.
		int numEntradas = 4;// dados de entrada (features)
		int numSaidas = 3;// rótulos (classes)
		double[][] dados = ged.dadosParaDouble(iris);

		// Gerando o dataset
		DataLoader loader = jnn.dataloader(dados, numEntradas, numSaidas);
		loader.embaralhar();
		DataLoader[] ds = loader.separar(0.75, 0.25);// separando 75% treino, 25% teste
		DataLoader treino = ds[0];
		
		// Criando um modelo
		Sequencial modelo = new Sequencial(
			new Entrada(numEntradas),
			new Densa(10, "tanh"),
			new Dropout(0.25),
			new Densa(10, "tanh"),
			new Dropout(0.25),
			new Densa(numSaidas, "softmax")
		);
			
		modelo.compilar("adam", "entropia-cruzada");
		modelo.setHistorico(true);// guardar valores de perda do treino
		
		treino.print();
		modelo.print();

		modelo.treinar(treino, 500, 12, true);
		
		// Avaliando o modelo
		DataLoader teste  = ds[1];
		System.out.println("Perda = " + modelo.avaliar(teste).item());
		double acc = modelo.avaliador().acuracia(teste).item();
		System.out.println("Acurácia = " + formatarDecimal(acc*100, 4) + "%");

		// Matriz de confusão
		modelo
		.avaliador()
		.matrizConfusao(teste)
		.nome("Matriz de confusão")
		.print();

		String nomeArquivo = "historico-perda.csv";
		exportarHistorico(modelo.hist(), nomeArquivo);

		try {
			executarComando("python grafico.py " + nomeArquivo);
		} catch (Exception e) {
			// 
		}
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
