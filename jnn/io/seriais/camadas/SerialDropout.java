package jnn.io.seriais.camadas;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.Dropout;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerializadorCamada;

/**
 * Interface de IO para camada Dropout.
 * @see jnn.camadas.Dropout
 */
class SerialDropout extends SerialBase implements SerializadorCamada<Dropout> {

	/**
	 * Interface de IO para camada MaxPool2D.
	 * @see jnn.camadas.Dropout
	 */
	public SerialDropout() {}
	
	@Deprecated
	public void serializar(Dropout camada, DataOutputStream dos) throws IOException {
		escrever(dos, camada.nome());

		int[] shapeIn = camada.shapeIn();
		escrever(dos, shapeIn);

		escrever(dos, camada.taxa());
	}

	@Override
	public Dropout ler(DataInputStream dis) throws IOException {
		int[] shapeIn = lerArrInt(dis);
		float taxa = lerFloat(dis);

		Dropout camada = new Dropout(taxa);
		camada.construir(shapeIn);
		
		return camada;
	}

	@Override
	public String nome() {
		return "dropout";
	}

	@Override
	public Class<Dropout> tipo() {
		return Dropout.class;
	}
}
