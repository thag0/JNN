package jnn.serial;

import java.io.BufferedReader;

import jnn.camadas.pooling.MaxPool2D;

class SerialMaxPool {

	/**
	 * Transforma os dados contidos na camada MaxPooling em
	 * informações sequenciais. Essas informações contém:
	 * <ul>
	 *    <li> Nome da camada; </li>
	 *    <li> Formato de entrada (altura, largura, profundidade); </li>
	 *    <li> Formato de saída (altura, largura, profundidade); </li>
	 *    <li> Formato do filtro (altura, largura); </li>
	 *    <li> Formato dos strides (altura, largura); </li>
	 * </ul>
	 * @param camada camada de max pooling que será serializada.
	 * @param sb StringBuilder usado como buffer.
	 */
	public void serializar(MaxPool2D camada, StringBuilder sb){
		//nome da camada pra facilitar
		sb.append(camada.nome()).append("\n");

		//formato de entrada
		int[] entrada = camada.shapeIn();
		for(int i = 0; i < entrada.length; i++){
			sb.append(entrada[i]).append(" ");
		}
		sb.append("\n");
		
		//formato de saída
		int[] saida = camada.shapeOut();
		for(int i = 0; i < saida.length; i++){
			sb.append(saida[i]).append(" ");
		}
		sb.append("\n");
		
		//formato do filtro
		int[] formFiltro = camada.formatoFiltro();
		for(int i = 0; i < formFiltro.length; i++){
			sb.append(formFiltro[i]).append(" ");
		}
		sb.append("\n");
		
		//formato dos strides
		int[] formStride = camada.formatoStride();
		for(int i = 0; i < formStride.length; i++){
			sb.append(formStride[i]).append(" ");
		}
		sb.append("\n");
	}

	/**
	 * Lê as informações da camada contida no arquivo.
	 * @param br leitor de buffer.
	 * @return instância de uma camada max pooling.
	 */
	public MaxPool2D lerConfig(BufferedReader br){
		try {
			//formato de entrada
			String[] sEntrada = br.readLine().split(" ");
			int[] entrada = new int[sEntrada.length];
			for (int i = 0; i < sEntrada.length; i++){ 
				entrada[i] = Integer.parseInt(sEntrada[i]);
			}

			//formato de saída
			String[] sSaida = br.readLine().split(" ");
			int[] saida = new int[sSaida.length];
			for (int i = 0; i < sSaida.length; i++) {
				saida[i] = Integer.parseInt(sSaida[i]);
			}

			//formato do filtro
			String[] sFiltro = br.readLine().split(" ");
			int[] filtro = new int[sFiltro.length];
			for (int i = 0; i < sFiltro.length; i++) {
				filtro[i] = Integer.parseInt(sFiltro[i]);
			}

			//formato dos strides
			String[] sStrides = br.readLine().split(" ");
			int[] strides = new int[sStrides.length];
			for (int i = 0; i < sStrides.length; i++) {
				strides[i] = Integer.parseInt(sStrides[i]);
			}

			MaxPool2D camada = new MaxPool2D(filtro, strides);
			camada.construir(entrada);
			return camada;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
}
