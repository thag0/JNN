package jnn.io.seriais.camadas;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.Flatten;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerializadorCamada;

/**
 * Interface de IO para camada Flatten.
 * @see jnn.camadas.Flatten
 */
class SerialFlatten extends SerialBase implements SerializadorCamada<Flatten> {

	/**
	 * Interface de IO para camada Flatten.
	 * @see jnn.camadas.Flatten
	 */
	public SerialFlatten() {}

	@Override
	public void serializar(Flatten camada, DataOutputStream dos) throws IOException {
		escrever(dos, camada.nome());
		
		int[] shapeIn = camada.shapeIn();
		escrever(dos, shapeIn);

		int[] shapeOut = camada.shapeOut();
		escrever(dos, shapeOut);
	}

	@Override
	@SuppressWarnings("unused")//só pro vscode não ficar reclamando
	public Flatten ler(DataInputStream dis) throws IOException {
		int[] shapeIn = lerArrInt(dis);
		int[] shapeOut = lerArrInt(dis);// tem que ser lido pra avançar o buffer

		Flatten camada = new Flatten();
		camada.construir(shapeIn);

		return camada;
	}

	@Override
	public String nome() {
		return "flatten";
	}

	@Override
	public Class<Flatten> tipo() {
		return Flatten.class;
	}
}
