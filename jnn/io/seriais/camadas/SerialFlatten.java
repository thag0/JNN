package jnn.io.seriais.camadas;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerialLayer;
import jnn.layers.Flatten;

/**
 * Interface de IO para camada Flatten.
 * @see jnn.layers.Flatten
 */
class SerialFlatten extends SerialBase implements SerialLayer<Flatten> {

	/**
	 * Interface de IO para camada Flatten.
	 * @see jnn.layers.Flatten
	 */
	public SerialFlatten() {}

	@Override
	public void serialize(Flatten layer, DataOutputStream dos) throws IOException {
		write(dos, layer.name());
		
		int[] shapeIn = layer.shapeIn();
		write(dos, shapeIn);

		int[] shapeOut = layer.shapeOut();
		write(dos, shapeOut);
	}

	@Override
	@SuppressWarnings("unused")//só pro vscode não ficar reclamando
	public Flatten read(DataInputStream dis) throws IOException {
		int[] shapeIn = readIntArray(dis);
		int[] shapeOut = readIntArray(dis);// tem que ser lido pra avançar o buffer

		Flatten layer = new Flatten();
		layer.build(shapeIn);

		return layer;
	}

	@Override
	public String name() {
		return "flatten";
	}

	@Override
	public Class<Flatten> type() {
		return Flatten.class;
	}
}
