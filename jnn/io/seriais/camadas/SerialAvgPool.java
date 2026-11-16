package jnn.io.seriais.camadas;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerialLayer;
import jnn.layers.pooling.AvgPool2D;

/**
 * Interface de IO para camada MaxPool2D.
 * @see jnn.layers.pooling.AvgPool2D
 */
class SerialAvgPool extends SerialBase implements SerialLayer<AvgPool2D> {

	/**
	 * Interface de IO para camada MaxPool2D.
	 * @see jnn.layers.pooling.AvgPool2D
	 */
	public SerialAvgPool() {}

	@Override
	public void serialize(AvgPool2D layer, DataOutputStream dos) throws IOException {
		write(dos, layer.name());
		
		int[] shapeIn = layer.shapeIn();
		write(dos, shapeIn);

		int[] shapeOut = layer.shapeOut();
		write(dos, shapeOut);

		int[] shapeKernel = layer.shapeFiltro();
		write(dos, shapeKernel);
		
		int[] shapeStride = layer.shapeStride();
		write(dos, shapeStride);
	}

	@Override
	@SuppressWarnings("unused")
	public AvgPool2D read(DataInputStream dis) throws IOException {
		// nome já é lido pra saber que camada é
		int[] shapeIn = readIntArray(dis);
		int[] shapeOut = readIntArray(dis);
		int[] shapeKernel = readIntArray(dis);
		int[] shapeStrides = readIntArray(dis);
	
		AvgPool2D layer = new AvgPool2D(shapeKernel, shapeStrides);
		layer.build(shapeIn);

		return layer;
	}

	@Override
	public String name() {
		return "avgpool2d";
	}

	@Override
	public Class<AvgPool2D> type() {
		return AvgPool2D.class;
	}
}
