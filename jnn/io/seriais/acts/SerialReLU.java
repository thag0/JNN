package jnn.io.seriais.acts;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerialLayer;
import jnn.layers.acts.ReLU;

/**
 * Serializados para a camada de ativação ReLU.
 */
public class SerialReLU extends SerialBase implements SerialLayer<ReLU> {
    
	/**
	 * Interface de IO para camada de ativação.
	 */
	public SerialReLU() {}

	@Override
	public void serialize(ReLU layer, DataOutputStream dos) throws IOException {
        write(dos, layer.name());
        write(dos, layer.shapeIn());
    }

	@Override
	public ReLU read(DataInputStream dis) throws IOException {
        int[] shapeIn = readIntArray(dis);
        return new ReLU(shapeIn);
    }

	@Override
	public String name() {
		return "relu";
	}

	@Override
	public Class<ReLU> type() {
		return ReLU.class;
	}

}
