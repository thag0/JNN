package jnn.io.seriais.acts;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.acts.Sigmoid;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerializadorCamada;

/**
 * Serializados para a camada de ativação Sigmoid.
 */
public class SerialSigmoid extends SerialBase implements SerializadorCamada<Sigmoid> {

	/**
	 * Interface de IO para camada de ativação.
	 */
	public SerialSigmoid() {}

	@Override
	public void serializar(Sigmoid camada, DataOutputStream dos) throws IOException {
        escrever(dos, camada.nome());
        escrever(dos, camada.shapeIn());
    }

	@Override
	public Sigmoid ler(DataInputStream dis) throws IOException {
        int[] shapeIn = lerArrInt(dis);
        return new Sigmoid(shapeIn);
    }

	@Override
	public String nome() {
		return "sigmoid";
	}

	@Override
	public Class<Sigmoid> tipo() {
		return Sigmoid.class;
	}

}
