package jnn.io.seriais.camadas;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.core.Parametro;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerialLayer;
import jnn.layers.Dense;

/**
 * Interface de IO para camada Densa.
 * @see jnn.layers.Dense
 */
class SerialDense extends SerialBase implements SerialLayer<Dense> {

	/**
	 * Interface de IO para camada Densa.
	 * @see jnn.layers.Dense
	 */
	public SerialDense() {}

	@Override
	public void serialize(Dense layer, DataOutputStream dos) throws IOException {
		write(dos, layer.name());

        int[] shapeIn = layer.shapeIn();
		write(dos, shapeIn);

        int[] shapeOut = layer.shapeOut();
		write(dos, shapeOut);
	
		write(dos, layer.hasBias());
		
		Parametro[] params = layer.params();

		float[] kernel = params[0].weight.data().toArray();
		write(dos, kernel);

		if (layer.hasBias()) {
			float[] bias = params[1].weight.data().toArray();
			write(dos, bias);
		}
	}

	@Override
	public Dense read(DataInputStream dis) throws IOException {
		// nome já é lido pra saber que camada é
		int[] shapeIn = readIntArray(dis);
		int[] shapeOut = readIntArray(dis);
		boolean hasBias = readBoolean(dis);
		float[] kernel = readFloatArray(dis);
		float[] bias = null;
		if (hasBias) bias = readFloatArray(dis);

		Dense layer = new Dense(shapeOut[0]);
		layer.setBias(hasBias);
		layer.build(shapeIn);
		
		Parametro[] params = layer.params();
		params[0].weight.copyElements(kernel);
		if (hasBias) params[1].weight.copyElements(bias);

		return layer;
	}

	@Override
	public String name() {
		return "dense";
	}

	@Override
	public Class<Dense> type() {
		return Dense.class;
	}

}
