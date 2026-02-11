package jnn.io.seriais.camadas;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.pooling.MaxPool2D;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerializadorCamada;

/**
 * Interface de IO para camada MaxPool2D.
 * @see jnn.camadas.pooling.MaxPool2D
 */
class SerialMaxPool extends SerialBase implements SerializadorCamada<MaxPool2D> {

	/**
	 * Interface de IO para camada MaxPool2D.
	 * @see jnn.camadas.pooling.MaxPool2D
	 */
	public SerialMaxPool() {}

	@Override
	public void serializar(MaxPool2D camada, DataOutputStream dos) throws IOException {
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

	@Override
	@SuppressWarnings("unused")// pro vscode não reclamar
	public MaxPool2D ler(DataInputStream dis) throws IOException {
		// nome já é lido pra saber que camada é
		int[] shapeIn = lerArrInt(dis);
		int[] shapeOut = lerArrInt(dis);
		int[] shapeFiltro = lerArrInt(dis);
		int[] shapeStrides = lerArrInt(dis);
	
		MaxPool2D camada = new MaxPool2D(shapeFiltro, shapeStrides);
		camada.construir(shapeIn);

		return camada;
	}

	@Override
	public String nome() {
		return "maxpool2d";
	}

	@Override
	public Class<MaxPool2D> tipo() {
		return MaxPool2D.class;
	}
}
