package jnn.io.seriais.camadas;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.Conv2D;
import jnn.core.Parametro;
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
		escrever(dos, camada.name());

		int[] shapeIn = camada.shapeIn();
		escrever(dos, shapeIn);

        int[] shapeOut = camada.shapeOut();
		escrever(dos, shapeOut);

		int[] filtro = camada.shapeKernel();
		escrever(dos, filtro);

		int[] padding = camada.shapePadding();
		escrever(dos, padding);
	
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
	public Conv2D ler(DataInputStream dis) throws IOException {
		// nome já é lido pra saber que camada é
		int[] shapeIn = lerArrInt(dis);
		int[] shapeOut = lerArrInt(dis);
		int[] filtro = lerArrInt(dis);
		int[] padding = lerArrInt(dis);
		boolean temBias = lerBoolean(dis);
		int numFiltros = shapeOut[0];
		int pad = padding[0];// altura == largura

		float[] kernel = lerArrFloat(dis);
		float[] bias = null;
		if (temBias) bias = lerArrFloat(dis);
		
		Conv2D camada = new Conv2D(numFiltros, filtro, pad);
		camada.setBias(temBias);
		camada.build(shapeIn);

		Parametro[] params = camada.params();
		
		params[0].weight.copyElements(kernel);

		if (temBias) params[1].weight.copyElements(bias);

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
