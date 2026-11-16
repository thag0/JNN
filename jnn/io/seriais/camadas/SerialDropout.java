package jnn.io.seriais.camadas;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerialLayer;
import jnn.layers.Dropout;

/**
 * Interface de IO para camada Dropout.
 * @see jnn.layers.Dropout
 */
class SerialDropout extends SerialBase implements SerialLayer<Dropout> {

	/**
	 * Interface de IO para camada MaxPool2D.
	 * @see jnn.layers.Dropout
	 */
	public SerialDropout() {}
	
	@Deprecated
	public void serialize(Dropout layer, DataOutputStream dos) throws IOException {
		write(dos, layer.name());

		int[] shapeIn = layer.shapeIn();
		write(dos, shapeIn);

		write(dos, layer.rate());
	}

	@Override
	public Dropout read(DataInputStream dis) throws IOException {
		int[] shapeIn = readIntArray(dis);
		float rate = readFloat(dis);

		Dropout layer = new Dropout(rate);
		layer.build(shapeIn);
		
		return layer;
	}

	@Override
	public String name() {
		return "dropout";
	}

	@Override
	public Class<Dropout> type() {
		return Dropout.class;
	}
}
