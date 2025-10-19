package exemplos.modelos;

import externos.lib.ged.Ged;
import jnn.camadas.Densa;
import jnn.modelos.Sequencial;
import jnn.serializacao.Serializador;

/**
 * Exemplo básico de serialização de um modelo em um arquivo externo.
 */
public class ExportarModelo {
	static Ged ged = new Ged();

	public static void main(String[] args){
		ged.limparConsole();

		// Caminho de destino do arquivo (deve conter a extensão ".nn")
		String caminho = "./dados/modelos/modelo.nn";

		// Criando um modelo base
		Sequencial modelo = new Sequencial(
			new Densa(784, "tanh"),
			new Densa(16, "tanh"),
			new Densa(16, "tanh"),
			new Densa(10, "sigmoid")
		);
		
		modelo.print();

		// Ferramenta para exportação
		Serializador serial = new Serializador();
		serial.salvar(modelo, caminho);
	}

}
