package jnn.io.seriais;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.Flatten;

/**
 * Interface de IO para camada Flatten.
 * @see jnn.camadas.Flatten
 */
class SerialFlatten extends SerialBase {

	/**
	 * Interface de IO para camada Flatten.
	 * @see jnn.camadas.Flatten
	 */
	public SerialFlatten() {}

	/**
	 * Transforma os dados da camada em uma estrutura sequencial. 
	 * @param camada camada base.
	 * @param dos {@code DataOutputStream} gravador.
     * @throws IOException caso ocorra um erro.
	 */
	public void serializar(Flatten camada, DataOutputStream dos) throws IOException {
		escrever(dos, camada.nome());
		
		int[] shapeIn = camada.shapeIn();
		escrever(dos, shapeIn);

		int[] shapeOut = camada.shapeOut();
		escrever(dos, shapeOut);
	}

	/**
	 * Lê as informações da camada.
	 * @param dis {@code DataInputStream} leitor.
	 * @return camada lida.
     * @throws IOException caso ocorra um erro.
	 */
	@SuppressWarnings("unused")//só pro vscode não ficar reclamando
	public Flatten ler(DataInputStream dis) throws IOException {
		int[] shapeIn = lerArrInt(dis);
		int[] shapeOut = lerArrInt(dis);// tem que ser lido pra avançar o buffer

		Flatten camada = new Flatten();
		camada.construir(shapeIn);

		return camada;
	}
}
