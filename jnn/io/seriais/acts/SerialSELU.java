package jnn.io.seriais.acts;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerialLayer;
import jnn.layers.acts.SELU;

/**
 * Serializados para a camada de ativação SELU.
 */
public class SerialSELU extends SerialBase implements SerialLayer<SELU> {

	/**
	 * Interface de IO para camada de ativação.
	 */
	public SerialSELU() {}

	@Override
	public void serialize(SELU layer, DataOutputStream dos) throws IOException {
        write(dos, layer.name());
        write(dos, layer.shapeIn());
        write(dos, layer.getAlpha());
        write(dos, layer.getGamma());
    }

	@Override
	public SELU read(DataInputStream dis) throws IOException {
        int[] shapeIn = readIntArray(dis);
        float alpha = readInt(dis);
        float gamma = readInt(dis);
        return new SELU(shapeIn, alpha, gamma);
    }

	@Override
	public String name() {
		return "selu";
	}

	@Override
	public Class<SELU> type() {
		return SELU.class;
	}

}
