package jnn.io.seriais.acts;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerialLayer;
import jnn.layers.acts.Swish;

/**
 * Serializados para a camada de ativação Swish.
 */
public class SerialSwish extends SerialBase implements SerialLayer<Swish> {

	/**
	 * Interface de IO para camada de ativação.
	 */
	public SerialSwish() {}

	@Override
	public void serialize(Swish layer, DataOutputStream dos) throws IOException {
        write(dos, layer.name());
        write(dos, layer.shapeIn());
    }

	@Override
	public Swish read(DataInputStream dis) throws IOException {
        int[] shapeIn = readIntArray(dis);
        return new Swish(shapeIn);
    }

	@Override
	public String name() {
		return "swish";
	}

	@Override
	public Class<Swish> type() {
		return Swish.class;
	}

}
