package jnn.io.seriais;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.Conv2D;

/**
 * Interface de IO para camada Conv2D.
 * @see jnn.camadas.Conv2D
 */
class SerialConv extends SerialBase {

	/**
	 * Interface de IO para camada Conv2D.
	 * @see jnn.camadas.Conv2D
	 */
	public SerialConv() {}

	/**
	 * Transforma os dados da camada em uma estrutura sequencial. 
	 * @param camada camada base.
	 * @param dos {@code DataOutputStream} gravador.
     * @throws IOException caso ocorra um erro.
	 */
	public void serializar(Conv2D camada, DataOutputStream dos) throws IOException {
		// //nome da camada pra facilitar
		// sb.append(camada.nome()).append("\n");

		// //formato de entrada
		// int[] entrada = camada.shapeIn();
		// for (int i = 0; i < entrada.length; i++) {
		// 	sb.append(entrada[i]).append(" ");
		// }
		// sb.append("\n");
		
		// //formato de saída
		// int[] saida = camada.shapeOut();
		// for (int i = 0; i < saida.length; i++) {
		// 	sb.append(saida[i]).append(" ");
		// }
		// sb.append("\n");
		
		// //formato dos filtros
		// int[] shapeFiltro = camada.formatoFiltro();
		// for (int i = 0; i < shapeFiltro.length; i++) {
		// 	sb.append(shapeFiltro[i]).append(" ");
		// }
		// sb.append("\n");
		
		// //função de ativação
		// sb.append(camada.ativacao().nome()).append("\n");

		// //bias
		// sb.append(camada.temBias()).append("\n");

		// //filtros
		// Tensor filtros = camada.kernel();
		// int[] shape = filtros.shape();
		// for (int i = 0; i < shape[0]; i++) {
		// 	for (int j = 0; j < shape[1]; j++) {
		// 		for (int k = 0; k < shape[2]; k++) {
		// 			for (int l = 0; l < shape[3]; l++) {
		// 				escreverDado(filtros.get(i, j, k, l), tipo, sb);
		// 				sb.append("\n");
		// 			}
		// 		}
		// 	}
		// }
		
		// if(camada.temBias()){
		// 	double[] bias = camada.bias().array();
		// 	for(double valor : bias){
		// 		escreverDado(valor, tipo, sb);
		// 		sb.append("\n");               
		// 	}
		// }
	
		escrever(dos, camada.nome());

		int[] shapeIn = camada.shapeIn();
		escrever(dos, shapeIn);

        int[] shapeOut = camada.shapeOut();
		escrever(dos, shapeOut);

		int[] filtro = camada.formatoFiltro();
		escrever(dos, filtro);

		escrever(dos, camada.ativacao().nome());
	
		escrever(dos, camada.temBias());

		double[] kernel = camada.kernel().data().paraArray();
		escrever(dos, kernel);

		if (camada.temBias()) {
			double[] bias = camada.bias().data().paraArray();
			escrever(dos, bias);
		}
	}

	/**
	 * Lê as informações da camada.
	 * @param dis {@code DataInputStream} leitor.
	 * @return camada lida.
     * @throws IOException caso ocorra um erro.
	 */
	public Conv2D ler(DataInputStream dis) throws IOException {
		// nome já é lido pra saber que camada é
		int[] shapeIn = lerArrInt(dis);
		int[] shapeOut = lerArrInt(dis);
		int[] filtro = lerArrInt(dis);
		String actStr = lerString(dis);
		boolean temBias = lerBoolean(dis);
		int numFiltros = shapeOut[0];

		double[] kernel = lerArrDouble(dis);
		double[] bias = null;
		if (temBias) bias = lerArrDouble(dis);
		
		Conv2D camada = new Conv2D(numFiltros, filtro);
		camada.setAtivacao(actStr);
		camada.setBias(temBias);
		camada.construir(shapeIn);

		camada.kernel().copiarElementos(kernel);
		if (temBias) camada.bias().copiarElementos(bias);

		return camada;
	}
}
