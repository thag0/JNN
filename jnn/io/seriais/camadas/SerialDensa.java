package jnn.io.seriais.camadas;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.Dense;
import jnn.core.Parametro;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerializadorCamada;

/**
 * Interface de IO para camada Densa.
 * @see jnn.camadas.Dense
 */
class SerialDensa extends SerialBase implements SerializadorCamada<Dense> {

	/**
	 * Interface de IO para camada Densa.
	 * @see jnn.camadas.Dense
	 */
	public SerialDensa() {}

	@Override
	public void serializar(Dense camada, DataOutputStream dos) throws IOException {
		escrever(dos, camada.name());

        int[] shapeIn = camada.shapeIn();
		escrever(dos, shapeIn);

        int[] shapeOut = camada.shapeOut();
		escrever(dos, shapeOut);
	
		escrever(dos, camada.hasBias());
		
		Parametro[] params = camada.params();

		float[] kernel = params[0].weight.data().toArray();
		escrever(dos, kernel);

		if (camada.hasBias()) {
			float[] bias = params[1].weight.data().toArray();
			escrever(dos, bias);
		}
	}

	@Override
	public Dense ler(DataInputStream dis) throws IOException {
		// nome já é lido pra saber que camada é
		int[] shapeIn = lerArrInt(dis);
		int[] shapeOut = lerArrInt(dis);
		boolean temBias = lerBoolean(dis);
		float[] kernel = lerArrFloat(dis);
		float[] bias = null;
		if (temBias) bias = lerArrFloat(dis);

		Dense camada = new Dense(shapeOut[0]);
		camada.setBias(temBias);
		camada.build(shapeIn);
		
		Parametro[] params = camada.params();
		params[0].weight.copyElements(kernel);
		if (temBias) params[1].weight.copyElements(bias);

		return camada;
	}

	@Override
	public String nome() {
		return "densa";
	}

	@Override
	public Class<Dense> tipo() {
		return Dense.class;
	}

}
