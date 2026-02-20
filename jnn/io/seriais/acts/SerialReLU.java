package jnn.io.seriais.acts;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.acts.ReLU;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerializadorCamada;

/**
 * Serializados para a camada de ativação ReLU.
 */
public class SerialReLU extends SerialBase implements SerializadorCamada<ReLU> {
    
	/**
	 * Interface de IO para camada de ativação.
	 */
	public SerialReLU() {}

	@Override
	public void serializar(ReLU camada, DataOutputStream dos) throws IOException {
        escrever(dos, camada.nome());
        escrever(dos, camada.shapeIn());
    }

	@Override
	public ReLU ler(DataInputStream dis) throws IOException {
        int[] shapeIn = lerArrInt(dis);
        return new ReLU(shapeIn);
    }

	@Override
	public String nome() {
		return "relu";
	}

	@Override
	public Class<ReLU> tipo() {
		return ReLU.class;
	}

}
