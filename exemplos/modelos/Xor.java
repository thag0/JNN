package exemplos.modelos;

import java.text.DecimalFormat;

import externos.lib.ged.Dados;
import externos.lib.ged.Ged;
import jnn.Funcional;
import jnn.camadas.*;
import jnn.core.tensor.Tensor;
import jnn.dataloader.DataLoader;
import jnn.modelos.*;
import jnn.otimizadores.SGD;

public class Xor{
	static Ged ged = new Ged();
	static Funcional jnn = new Funcional();

	public static void main(String[] args){
		ged.limparConsole();

		// Carregando dados e convertendo no DataLoader
		Dados xor = ged.lerCsv("./dados/csv/xor.csv");
		double[][] dados = ged.dadosParaDouble(xor);
		int in  = 2;// Entradas X
		int out = 1;// Saídas Y
		DataLoader dXor = jnn.dataloader(dados, in, out);

		// Criando de treinando o modelo
		Sequencial modelo = new Sequencial(
			new Entrada(in),
			new Densa(2, "sigmoid"),
			new Densa(out, "sigmoid")
		);
		modelo.compilar(new SGD(0.001, 0.999), "mse");
		modelo.treinar(dXor, 5_000, false);
		
		// Avaliando
		Tensor[] xs = dXor.getX();
		Tensor[] ys = dXor.getY();
		Tensor perda = modelo.avaliar(xs, ys);
		System.out.println("Perda: " + perda.item());

		// Criando uma tabela verdade para verificação
		verificar(modelo);
	}

	/**
	 * Validação do modelo xor usando uma tabela verdade.
	 * @param modelo modelo base.
	 */
	static void verificar(Sequencial modelo) {
		System.out.println("\n-- Tabela verdade --");
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				Tensor x = new Tensor(new double[]{ i, j });
				Tensor y = modelo.forward(x);
				System.out.println(i + " ^ " + j + " = " + formatarDecimal(y.item(), 10));
			}
		}
	}

	/**
	 * Formata o valor em uma String.
	 * @param valor valor base.
	 * @param casas casas decimais.
	 * @return {@code String} com valor formatado
	 */
	static String formatarDecimal(double valor, int casas) {
		String formato = "#." + "#".repeat(casas);
		String valStr = new DecimalFormat(formato).format(valor);
		return valStr.replaceAll(",", ".");
	}

}
