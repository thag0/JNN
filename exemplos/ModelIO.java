package exemplos;

import ged.Ged;
import jnn.camadas.Dense;
import jnn.io.JNNserial;
import jnn.modelos.Sequencial;

/**
 * Exemplo básico de serialização de um modelo em um arquivo externo.
 */
public class ModelIO {
	static Ged ged = new Ged();

	public static void main(String[] args){
		ged.limparConsole();

		// Caminho de destino do arquivo (deve conter a extensão ".nn")
		String caminho = "./dados/modelos/modelo.nn";

		// Criando um modelo base
		Sequencial modelo = new Sequencial(
			new Dense(784),
			new Dense(16),
			new Dense(16),
			new Dense(10)
		);
		
		modelo.print();

		// Ferramenta para exportação
		JNNserial.salvar(modelo, caminho);
	}

}
