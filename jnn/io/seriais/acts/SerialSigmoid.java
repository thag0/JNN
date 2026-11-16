package jnn.io.seriais.acts;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerialLayer;
import jnn.layers.acts.Sigmoid;

/**
 * Serializados para a camada de ativação Sigmoid.
 */
public class SerialSigmoid extends SerialBase implements SerialLayer<Sigmoid> {

	/**
	 * Interface de IO para camada de ativação.
	 */
	public SerialSigmoid() {}

	@Override
	public void serialize(Sigmoid layer, DataOutputStream dos) throws IOException {
        write(dos, layer.name());
        write(dos, layer.shapeIn());
    }

	@Override
	public Sigmoid read(DataInputStream dis) throws IOException {
        int[] shapeIn = readIntArray(dis);
        return new Sigmoid(shapeIn);
    }

	@Override
	public String name() {
		return "sigmoid";
	}

	@Override
	public Class<Sigmoid> type() {
		return Sigmoid.class;
	}

}
