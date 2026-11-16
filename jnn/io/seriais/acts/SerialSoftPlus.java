package jnn.io.seriais.acts;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerialLayer;
import jnn.layers.acts.Softplus;

/**
 * Serializados para a camada de ativação Softplus.
 */
public class SerialSoftPlus extends SerialBase implements SerialLayer<Softplus> {
    
	/**
	 * Interface de IO para camada de ativação.
	 */
	public SerialSoftPlus() {}

	@Override
	public void serialize(Softplus layer, DataOutputStream dos) throws IOException {
        write(dos, layer.name());
        write(dos, layer.shapeIn());
    }

	@Override
	public Softplus read(DataInputStream dis) throws IOException {
        int[] shapeIn = readIntArray(dis);
        return new Softplus(shapeIn);
    }

	@Override
	public String name() {
		return "softplus";
	}

	@Override
	public Class<Softplus> type() {
		return Softplus.class;
	}

}
