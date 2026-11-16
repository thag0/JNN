package jnn.io.seriais.camadas;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerialLayer;
import jnn.layers.pooling.MaxPool2D;

/**
 * Interface de IO para camada MaxPool2D.
 * @see jnn.layers.pooling.MaxPool2D
 */
class SerialMaxPool extends SerialBase implements SerialLayer<MaxPool2D> {

	/**
	 * Interface de IO para camada MaxPool2D.
	 * @see jnn.layers.pooling.MaxPool2D
	 */
	public SerialMaxPool() {}

	@Override
	public void serialize(MaxPool2D layer, DataOutputStream dos) throws IOException {
		write(dos, layer.name());
		
		int[] shapeIn = layer.shapeIn();
		write(dos, shapeIn);

		int[] shapeOut = layer.shapeOut();
		write(dos, shapeOut);

		int[] shapeFiltro = layer.shapeFiltro();
		write(dos, shapeFiltro);
		
		int[] shapeStride = layer.shapeStride();
		write(dos, shapeStride);
	}

	@Override
	@SuppressWarnings("unused")// pro vscode não reclamar
	public MaxPool2D read(DataInputStream dis) throws IOException {
		// nome já é lido pra saber que camada é
		int[] shapeIn = readIntArray(dis);
		int[] shapeOut = readIntArray(dis);
		int[] shapeFiltro = readIntArray(dis);
		int[] shapeStrides = readIntArray(dis);
	
		MaxPool2D layer = new MaxPool2D(shapeFiltro, shapeStrides);
		layer.build(shapeIn);

		return layer;
	}

	@Override
	public String name() {
		return "maxpool2d";
	}

	@Override
	public Class<MaxPool2D> type() {
		return MaxPool2D.class;
	}
}
