package jnn.io.seriais.acts;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.acts.Softmax;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerializadorCamada;

/**
 * Serializados para a camada de ativação Softmax.
 */
public class SerialSoftmax extends SerialBase implements SerializadorCamada<Softmax> {

	/**
	 * Interface de IO para camada de ativação.
	 */
	public SerialSoftmax() {}

	@Override
	public void serializar(Softmax camada, DataOutputStream dos) throws IOException {
        escrever(dos, camada.nome());
        escrever(dos, camada.shapeIn());
    }

	@Override
	public Softmax ler(DataInputStream dis) throws IOException {
        int[] shapeIn = lerArrInt(dis);
        return new Softmax(shapeIn);
    }

	@Override
	public String nome() {
		return "softmax";
	}

	@Override
	public Class<Softmax> tipo() {
		return Softmax.class;
	}

}
