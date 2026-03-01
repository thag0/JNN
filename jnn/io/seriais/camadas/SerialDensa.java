package jnn.io.seriais.camadas;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.Densa;
import jnn.core.tensor.Tensor;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerializadorCamada;

/**
 * Interface de IO para camada Densa.
 * @see jnn.camadas.Densa
 */
class SerialDensa extends SerialBase implements SerializadorCamada<Densa> {

	/**
	 * Interface de IO para camada Densa.
	 * @see jnn.camadas.Densa
	 */
	public SerialDensa() {}

	@Override
	public void serializar(Densa camada, DataOutputStream dos) throws IOException {
		escrever(dos, camada.nome());

        int[] shapeIn = camada.shapeIn();
		escrever(dos, shapeIn);

        int[] shapeOut = camada.shapeOut();
		escrever(dos, shapeOut);
	
		escrever(dos, camada.temBias());
		
		Tensor[] params = camada.params();

		float[] kernel = params[0].data().paraArray();
		escrever(dos, kernel);

		if (camada.temBias()) {
			float[] bias = params[1].data().paraArray();
			escrever(dos, bias);
		}
	}

	@Override
	public Densa ler(DataInputStream dis) throws IOException {
		// nome já é lido pra saber que camada é
		int[] shapeIn = lerArrInt(dis);
		int[] shapeOut = lerArrInt(dis);
		boolean temBias = lerBoolean(dis);
		float[] kernel = lerArrFloat(dis);
		float[] bias = null;
		if (temBias) bias = lerArrFloat(dis);

		Densa camada = new Densa(shapeOut[0]);
		camada.setBias(temBias);
		camada.construir(shapeIn);
		
		Tensor[] params = camada.params();
		params[0].copiarElementos(kernel);
		if (temBias) params[1].copiarElementos(bias);

		return camada;
	}

	@Override
	public String nome() {
		return "densa";
	}

	@Override
	public Class<Densa> tipo() {
		return Densa.class;
	}

}
