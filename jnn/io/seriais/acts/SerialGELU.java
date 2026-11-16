package jnn.io.seriais.acts;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerialLayer;
import jnn.layers.acts.GELU;

/**
 * Serializados para a camada de ativação GELU.
 */
public class SerialGELU extends SerialBase implements SerialLayer<GELU> {
    
	/**
	 * Interface de IO para camada de ativação.
	 */
	public SerialGELU() {}

	@Override
	public void serialize(GELU layer, DataOutputStream dos) throws IOException {
        write(dos, layer.name());
        write(dos, layer.shapeIn());
    }

	@Override
	public GELU read(DataInputStream dis) throws IOException {
        int[] shapeIn = readIntArray(dis);
        return new GELU(shapeIn);
    }

	@Override
	public String name() {
		return "gelu";
	}

	@Override
	public Class<GELU> type() {
		return GELU.class;
	}

}
