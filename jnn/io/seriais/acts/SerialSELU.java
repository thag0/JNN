package jnn.io.seriais.acts;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.acts.SELU;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerializadorCamada;

/**
 * Serializados para a camada de ativação SELU.
 */
public class SerialSELU extends SerialBase implements SerializadorCamada<SELU> {

	/**
	 * Interface de IO para camada de ativação.
	 */
	public SerialSELU() {}

	@Override
	public void serializar(SELU camada, DataOutputStream dos) throws IOException {
        escrever(dos, camada.nome());
        escrever(dos, camada.shapeIn());
        escrever(dos, camada.getAlpha());
        escrever(dos, camada.getGamma());
    }

	@Override
	public SELU ler(DataInputStream dis) throws IOException {
        int[] shapeIn = lerArrInt(dis);
        float alpha = lerInt(dis);
        float gamma = lerInt(dis);
        return new SELU(shapeIn, alpha, gamma);
    }

	@Override
	public String nome() {
		return "selu";
	}

	@Override
	public Class<SELU> tipo() {
		return SELU.class;
	}

}
