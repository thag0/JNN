package jnn.io.seriais.camadas;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.Conv2D;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerializadorCamada;

/**
 * Interface de IO para camada Conv2D.
 * @see jnn.camadas.Conv2D
 */
class SerialConv extends SerialBase implements SerializadorCamada<Conv2D> {

	/**
	 * Interface de IO para camada Conv2D.
	 * @see jnn.camadas.Conv2D
	 */
	public SerialConv() {}

	@Override
	public void serializar(Conv2D camada, DataOutputStream dos) throws IOException {
		escrever(dos, camada.nome());

		int[] shapeIn = camada.shapeIn();
		escrever(dos, shapeIn);

        int[] shapeOut = camada.shapeOut();
		escrever(dos, shapeOut);

		int[] filtro = camada.formatoFiltro();
		escrever(dos, filtro);
	
		escrever(dos, camada.temBias());

		float[] kernel = camada.kernel().data().paraArray();
		escrever(dos, kernel);

		if (camada.temBias()) {
			float[] bias = camada.bias().data().paraArray();
			escrever(dos, bias);
		}
	}

	@Override
	public Conv2D ler(DataInputStream dis) throws IOException {
		// nome já é lido pra saber que camada é
		int[] shapeIn = lerArrInt(dis);
		int[] shapeOut = lerArrInt(dis);
		int[] filtro = lerArrInt(dis);
		boolean temBias = lerBoolean(dis);
		int numFiltros = shapeOut[0];

		float[] kernel = lerArrFloat(dis);
		float[] bias = null;
		if (temBias) bias = lerArrFloat(dis);
		
		Conv2D camada = new Conv2D(numFiltros, filtro);
		camada.setBias(temBias);
		camada.construir(shapeIn);

		camada.kernel().copiarElementos(kernel);
		if (temBias) camada.bias().copiarElementos(bias);

		return camada;
	}

	@Override
	public String nome() {
		return "conv2d";
	}

	@Override
	public Class<Conv2D> tipo() {
		return Conv2D.class;
	}
}
