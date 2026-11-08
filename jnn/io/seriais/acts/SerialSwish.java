package jnn.io.seriais.acts;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerializadorCamada;
import jnn.layers.acts.Swish;

/**
 * Serializados para a camada de ativação Swish.
 */
public class SerialSwish extends SerialBase implements SerializadorCamada<Swish> {

	/**
	 * Interface de IO para camada de ativação.
	 */
	public SerialSwish() {}

	@Override
	public void serializar(Swish camada, DataOutputStream dos) throws IOException {
        escrever(dos, camada.name());
        escrever(dos, camada.shapeIn());
    }

	@Override
	public Swish ler(DataInputStream dis) throws IOException {
        int[] shapeIn = lerArrInt(dis);
        return new Swish(shapeIn);
    }

	@Override
	public String nome() {
		return "swish";
	}

	@Override
	public Class<Swish> tipo() {
		return Swish.class;
	}

}
