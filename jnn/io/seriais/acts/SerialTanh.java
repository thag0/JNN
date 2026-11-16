package jnn.io.seriais.acts;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerialLayer;
import jnn.layers.acts.Tanh;

/**
 * Serializados para a camada de ativação Tanh.
 */
public class SerialTanh extends SerialBase implements SerialLayer<Tanh> {
    
	/**
	 * Interface de IO para camada de ativação.
	 */
	public SerialTanh() {}

	@Override
	public void serialize(Tanh layer, DataOutputStream dos) throws IOException {
        write(dos, layer.name());
        write(dos, layer.shapeIn());
    }

	@Override
	public Tanh read(DataInputStream dis) throws IOException {
        int[] shapeIn = readIntArray(dis);
        return new Tanh(shapeIn);
    }

	@Override
	public String name() {
		return "tanh";
	}

	@Override
	public Class<Tanh> type() {
		return Tanh.class;
	}

}
