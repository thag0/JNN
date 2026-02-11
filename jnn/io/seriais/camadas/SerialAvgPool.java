package jnn.io.seriais.camadas;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.pooling.AvgPool2D;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerializadorCamada;

/**
 * Interface de IO para camada MaxPool2D.
 * @see jnn.camadas.pooling.AvgPool2D
 */
class SerialAvgPool extends SerialBase implements SerializadorCamada<AvgPool2D> {

	/**
	 * Interface de IO para camada MaxPool2D.
	 * @see jnn.camadas.pooling.AvgPool2D
	 */
	public SerialAvgPool() {}

	@Override
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

	@Override
	@SuppressWarnings("unused")
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

	@Override
	public String nome() {
		return "avgpool2d";
	}

	@Override
	public Class<AvgPool2D> tipo() {
		return AvgPool2D.class;
	}
}
