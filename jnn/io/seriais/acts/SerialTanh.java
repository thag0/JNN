package jnn.io.seriais.acts;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.acts.Tanh;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerializadorCamada;

/**
 * Serializados para a camada de ativação Tanh.
 */
public class SerialTanh extends SerialBase implements SerializadorCamada<Tanh> {
    
	/**
	 * Interface de IO para camada de ativação.
	 */
	public SerialTanh() {}

	@Override
	public void serializar(Tanh camada, DataOutputStream dos) throws IOException {
        escrever(dos, camada.nome());
        escrever(dos, camada.shapeIn());
    }

	@Override
	public Tanh ler(DataInputStream dis) throws IOException {
        int[] shapeIn = lerArrInt(dis);
        return new Tanh(shapeIn);
    }

	@Override
	public String nome() {
		return "tanh";
	}

	@Override
	public Class<Tanh> tipo() {
		return Tanh.class;
	}

}
