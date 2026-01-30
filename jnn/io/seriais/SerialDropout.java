package jnn.io.seriais;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.Dropout;

/**
 * Interface de IO para camada Dropout.
 * @see jnn.camadas.Dropout
 */
class SerialDropout extends SerialBase {

	/**
	 * Interface de IO para camada MaxPool2D.
	 * @see jnn.camadas.Dropout
	 */
	public SerialDropout() {}
	
	/**
	 * Transforma os dados da camada em uma estrutura sequencial. 
	 * @param camada camada base.
	 * @param dos {@code DataOutputStream} gravador.
     * @throws IOException caso ocorra um erro.
	 */
	public void serializar(Dropout camada, DataOutputStream dos) throws IOException {
		escrever(dos, camada.nome());

		int[] shapeIn = camada.shapeIn();
		escrever(dos, shapeIn);

		escrever(dos, camada.taxa());
	}

	/**
	 * Lê as informações da camada contida no arquivo.
	 * @param br leitor de buffer.
	 * @return instância de uma camada dropout.
	 */
	public Dropout ler(DataInputStream dis) throws IOException {
		int[] shapeIn = lerArrInt(dis);
		float taxa = lerFloat(dis);

		Dropout camada = new Dropout(taxa);
		camada.construir(shapeIn);
		
		return camada;
	}
}
