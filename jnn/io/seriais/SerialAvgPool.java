package jnn.io.seriais;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.pooling.AvgPool2D;

/**
 * Interface de IO para camada MaxPool2D.
 * @see jnn.camadas.pooling.AvgPool2D
 */
class SerialAvgPool extends SerialBase {

	/**
	 * Interface de IO para camada MaxPool2D.
	 * @see jnn.camadas.pooling.AvgPool2D
	 */
	public SerialAvgPool() {}

	/**
	 * Transforma os dados da camada em uma estrutura sequencial. 
	 * @param camada camada base.
	 * @param dos {@code DataOutputStream} gravador.
     * @throws IOException caso ocorra um erro.
	 */
	public void serializar(AvgPool2D camada, DataOutputStream dos) throws IOException {
		escrever(dos, camada.nome());
		
		int[] shapeIn = camada.shapeIn();
		escrever(dos, shapeIn);

		int[] shapeOut = camada.shapeOut();
		escrever(dos, shapeOut);

		int[] shapeFiltro = camada.formatoFiltro();
		escrever(dos, shapeFiltro);
		
		int[] shapeStride = camada.formatoStride();
		escrever(dos, shapeStride);
	}

	/**
	 * Lê as informações da camada.
	 * @param dis {@code DataInputStream} leitor.
	 * @return camada lida.
     * @throws IOException caso ocorra um erro.
	 */
	@SuppressWarnings("unused")// pro vscode não reclamar
	public AvgPool2D ler(DataInputStream dis) throws IOException {
		// nome já é lido pra saber que camada é
		int[] shapeIn = lerArrInt(dis);
		int[] shapeOut = lerArrInt(dis);
		int[] shapeFiltro = lerArrInt(dis);
		int[] shapeStrides = lerArrInt(dis);
	
		AvgPool2D camada = new AvgPool2D(shapeFiltro, shapeStrides);
		camada.construir(shapeIn);

		return camada;
	}
}
