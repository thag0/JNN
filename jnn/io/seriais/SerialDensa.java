package jnn.io.seriais;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.Densa;

/**
 * Interface de IO para camada Densa.
 * @see jnn.camadas.Densa
 */
class SerialDensa extends SerialBase {

	/**
	 * Interface de IO para camada Densa.
	 * @see jnn.camadas.Densa
	 */
	public SerialDensa() {}

	/**
	 * Transforma os dados da camada em uma estrutura sequencial. 
	 * @param camada camada base.
	 * @param dos {@code DataOutputStream} gravador.
     * @throws IOException caso ocorra um erro.
	 */
	public void serializar(Densa camada, DataOutputStream dos) throws IOException {
		escrever(dos, camada.nome());

        int[] shapeIn = camada.shapeIn();
		escrever(dos, shapeIn);

        int[] shapeOut = camada.shapeOut();
		escrever(dos, shapeOut);
	
		escrever(dos, camada.ativacao().nome());
	
		escrever(dos, camada.temBias());
		
		float[] kernel = camada.kernel().data().paraArray();
		escrever(dos, kernel);

		if (camada.temBias()) {
			float[] bias = camada.bias().data().paraArray();
			escrever(dos, bias);
		}
	}

	/**
	 * Lê as informações da camada.
	 * @param dis {@code DataInputStream} leitor.
	 * @return camada lida.
     * @throws IOException caso ocorra um erro.
	 */
	public Densa ler(DataInputStream dis) throws IOException {
		// nome já é lido pra saber que camada é
		int[] shapeIn = lerArrInt(dis);
		int[] shapeOut = lerArrInt(dis);
		String actStr = lerString(dis);
		boolean temBias = lerBoolean(dis);
		float[] kernel = lerArrFloat(dis);
		float[] bias = null;
		if (temBias) bias = lerArrFloat(dis);

		Densa camada = new Densa(shapeOut[0], actStr);
		camada.setBias(temBias);
		camada.construir(shapeIn);
		
		camada.kernel().copiarElementos(kernel);
		if (temBias) camada.bias().copiarElementos(bias);

		return camada;
	}

}
