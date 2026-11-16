package jnn.io.seriais.acts;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerialLayer;
import jnn.layers.acts.Softmax;

/**
 * Serializados para a camada de ativação Softmax.
 */
public class SerialSoftmax extends SerialBase implements SerialLayer<Softmax> {

	/**
	 * Interface de IO para camada de ativação.
	 */
	public SerialSoftmax() {}

	@Override
	public void serialize(Softmax layer, DataOutputStream dos) throws IOException {
        write(dos, layer.name());
        write(dos, layer.shapeIn());
    }

	@Override
	public Softmax read(DataInputStream dis) throws IOException {
        int[] shapeIn = readIntArray(dis);
        return new Softmax(shapeIn);
    }

	@Override
	public String name() {
		return "softmax";
	}

	@Override
	public Class<Softmax> type() {
		return Softmax.class;
	}

}
