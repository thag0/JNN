package jnn.io.seriais.acts;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.acts.GELU;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerializadorCamada;

/**
 * Serializados para a camada de ativação GELU.
 */
public class SerialGELU extends SerialBase implements SerializadorCamada<GELU> {
    
	/**
	 * Interface de IO para camada de ativação.
	 */
	public SerialGELU() {}

	@Override
	public void serializar(GELU camada, DataOutputStream dos) throws IOException {
        escrever(dos, camada.nome());
        escrever(dos, camada.shapeIn());
    }

	@Override
	public GELU ler(DataInputStream dis) throws IOException {
        int[] shapeIn = lerArrInt(dis);
        return new GELU(shapeIn);
    }

	@Override
	public String nome() {
		return "gelu";
	}

	@Override
	public Class<GELU> tipo() {
		return GELU.class;
	}

}
