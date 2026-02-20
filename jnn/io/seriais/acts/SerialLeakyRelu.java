package jnn.io.seriais.acts;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.acts.LeakyReLU;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerializadorCamada;

/**
 * Serializados para a camada de ativação Leaky ReLU.
 */
public class SerialLeakyRelu extends SerialBase implements SerializadorCamada<LeakyReLU> {

	/**
	 * Interface de IO para camada de ativação.
	 */
	public SerialLeakyRelu() {}

	@Override
	public void serializar(LeakyReLU camada, DataOutputStream dos) throws IOException {
        escrever(dos, camada.nome());
        escrever(dos, camada.shapeIn());
        escrever(dos, camada.getAlpha());
    }

	@Override
	public LeakyReLU ler(DataInputStream dis) throws IOException {
        int[] shapeIn = lerArrInt(dis);
        float alpha = lerFloat(dis);
        return new LeakyReLU(alpha, shapeIn);
    }

	@Override
	public String nome() {
		return "leakyrelu";
	}

	@Override
	public Class<LeakyReLU> tipo() {
		return LeakyReLU.class;
	}

}
