package jnn.io.seriais.acts;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerialLayer;
import jnn.layers.acts.LeakyReLU;

/**
 * Serializados para a camada de ativação Leaky ReLU.
 */
public class SerialLeakyRelu extends SerialBase implements SerialLayer<LeakyReLU> {

	/**
	 * Interface de IO para camada de ativação.
	 */
	public SerialLeakyRelu() {}

	@Override
	public void serialize(LeakyReLU layer, DataOutputStream dos) throws IOException {
        write(dos, layer.name());
        write(dos, layer.shapeIn());
        write(dos, layer.getAlpha());
    }

	@Override
	public LeakyReLU read(DataInputStream dis) throws IOException {
        int[] shapeIn = readIntArray(dis);
        float alpha = readFloat(dis);
        return new LeakyReLU(alpha, shapeIn);
    }

	@Override
	public String name() {
		return "leakyrelu";
	}

	@Override
	public Class<LeakyReLU> type() {
		return LeakyReLU.class;
	}

}
