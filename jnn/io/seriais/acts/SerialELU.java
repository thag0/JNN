package jnn.io.seriais.acts;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerialLayer;
import jnn.layers.acts.ELU;

/**
 * Serializados para a camada de ativação ELU.
 */
public class SerialELU extends SerialBase implements SerialLayer<ELU> {

	/**
	 * Interface de IO para camada de ativação.
	 */
	public SerialELU() {}

	@Override
	public void serialize(ELU layer, DataOutputStream dos) throws IOException {
        write(dos, layer.name());
        write(dos, layer.shapeIn());
        write(dos, layer.getAlpha());
    }

	@Override
	public ELU read(DataInputStream dis) throws IOException {
        int[] shapeIn = readIntArray(dis);
        float alpha = readFloat(dis);
        return new ELU(alpha, shapeIn);
    }

	@Override
	public String name() {
		return "elu";
	}

	@Override
	public Class<ELU> type() {
		return ELU.class;
	}

}
