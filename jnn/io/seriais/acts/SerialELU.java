package jnn.io.seriais.acts;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.acts.ELU;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerializadorCamada;

/**
 * Serializados para a camada de ativação ELU.
 */
public class SerialELU extends SerialBase implements SerializadorCamada<ELU> {

	/**
	 * Interface de IO para camada de ativação.
	 */
	public SerialELU() {}

	@Override
	public void serializar(ELU camada, DataOutputStream dos) throws IOException {
        escrever(dos, camada.nome());
        escrever(dos, camada.shapeIn());
        escrever(dos, camada.getAlpha());
    }

	@Override
	public ELU ler(DataInputStream dis) throws IOException {
        int[] shapeIn = lerArrInt(dis);
        float alfa = lerFloat(dis);
        return new ELU(alfa, shapeIn);
    }

	@Override
	public String nome() {
		return "elu";
	}

	@Override
	public Class<ELU> tipo() {
		return ELU.class;
	}

}
