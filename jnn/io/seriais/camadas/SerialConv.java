
package jnn.io.seriais.camadas;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.core.Parametro;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerialLayer;
import jnn.layers.Conv2D;

/**
 * Interface de IO para camada Conv2D.
 * @see jnn.layers.Conv2D
 */
class SerialConv extends SerialBase implements SerialLayer<Conv2D> {

	/**
	 * Interface de IO para camada Conv2D.
	 * @see jnn.layers.Conv2D
	 */
	public SerialConv() {}

	@Override
	public void serialize(Conv2D layer, DataOutputStream dos) throws IOException {
		write(dos, layer.name());

		int[] shapeIn = layer.shapeIn();
		write(dos, shapeIn);

        int[] shapeOut = layer.shapeOut();
		write(dos, shapeOut);

		int[] shapeKernel = layer.shapeKernel();
		write(dos, shapeKernel);

		int[] shapePadding = layer.shapePadding();
		write(dos, shapePadding);
	
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
	public Conv2D read(DataInputStream dis) throws IOException {
		// nome já é lido pra saber que camada é
		int[] shapeIn = readIntArray(dis);
		int[] shapeOut = readIntArray(dis);
		int[] shapeKernel = readIntArray(dis);
		int[] shapePadding = readIntArray(dis);
		boolean hasBias = readBoolean(dis);
		int numFilters = shapeOut[0];
		int padding = shapePadding[0];// altura == largura

		float[] kernel = readFloatArray(dis);
		float[] bias = null;
		if (hasBias) bias = readFloatArray(dis);
		
		Conv2D layer = new Conv2D(numFilters, shapeKernel, padding);
		layer.setBias(hasBias);
		layer.build(shapeIn);

		Parametro[] params = layer.params();
		params[0].weight.copyElements(kernel);
		if (hasBias) params[1].weight.copyElements(bias);

		return layer;
	}

	@Override
	public String name() {
		return "conv2d";
	}

	@Override
	public Class<Conv2D> type() {
		return Conv2D.class;
	}
}
