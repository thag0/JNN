package jnn.io.seriais.acts;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerializadorCamada;
import jnn.layers.acts.Softplus;

/**
 * Serializados para a camada de ativação Softplus.
 */
public class SerialSoftPlus extends SerialBase implements SerializadorCamada<Softplus> {
    
	/**
	 * Interface de IO para camada de ativação.
	 */
	public SerialSoftPlus() {}

	@Override
	public void serializar(Softplus camada, DataOutputStream dos) throws IOException {
        escrever(dos, camada.name());
        escrever(dos, camada.shapeIn());
    }

	@Override
	public Softplus ler(DataInputStream dis) throws IOException {
        int[] shapeIn = lerArrInt(dis);
        return new Softplus(shapeIn);
    }

	@Override
	public String nome() {
		return "softplus";
	}

	@Override
	public Class<Softplus> tipo() {
		return Softplus.class;
	}

}
