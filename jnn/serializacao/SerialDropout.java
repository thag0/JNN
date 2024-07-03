package jnn.serializacao;

import java.io.BufferedReader;

import jnn.camadas.Dropout;

class SerialDropout {
	
	/**
	 * Transforma os dados contidos na camada Dropou numa sequência
	 * de informações sequenciais. Essas informações contém:
	 * <ul>
	 *    <li> Nome da camada; </li>
	 *    <li> Formato de entrada (altura, largura, profundidade); </li>
	 *    <li> taxa de dropout; </li>
	 * </ul>
	 * @param camada camada de dropout que será serializada.
	 * @param sb StringBuilder usado como buffer.
	 */
	public void serializar(Dropout camada, StringBuilder sb) {
		//nome da camada pra facilitar
		sb.append(camada.nome()).append("\n");

		//formato de entrada
		int[] entrada = camada.shapeEntrada();
		for (int i = 0; i < entrada.length; i++) {
			sb.append(entrada[i]).append(" ");
		}
		sb.append("\n");
		
		//taxa
		sb.append(camada.taxa()).append("\n");
	}

	/**
	 * Lê as informações da camada contida no arquivo.
	 * @param br leitor de buffer.
	 * @return instância de uma camada dropout.
	 */
	public Dropout lerConfig(BufferedReader br) {
		try {
			//formato de entrada
			String[] sEntrada = br.readLine().split(" ");
			int[] entrada = new int[sEntrada.length];
			for (int i = 0; i < sEntrada.length; i++) {
				entrada[i] = Integer.parseInt(sEntrada[i]);
			}

			//taxa
			double taxa = Double.parseDouble(br.readLine());

			Dropout camada = new Dropout(taxa);
			camada.construir(entrada);
			return camada;
		
		} catch(Exception e) {
			throw new RuntimeException(e);
		}
	}
}
