package jnn.serializacao;

import java.io.BufferedReader;
import java.io.BufferedWriter;

import jnn.camadas.Flatten;

class SerialFlatten {

	/**
	 * Transforma os dados contidos na camada Flatten numa sequência
	 * de informações sequenciais. Essas informações contém:
	 * <ul>
	 *    <li> Nome da camada; </li>
	 *    <li> Formato de entrada (altura, largura, profundidade); </li>
	 *    <li> Formato de saída (altura, largura, profundidade); </li>
	 * </ul>
	 * @param camada camada flatten que será serializada.
	 * @param bw escritor de buffer usado para salvar os dados da camada.
	 */
	public void serializar(Flatten camada, BufferedWriter bw) {
		try{
			//nome da camada pra facilitar
			bw.write(camada.nome());
			bw.newLine();

			//formato de entrada
			int[] entrada = camada.formatoEntrada();
			for (int i = 0; i < entrada.length; i++) {
				bw.write(entrada[i] + " ");
			}
			bw.newLine();
			
			//formato de saída
			int[] saida = camada.formatoSaida();
			for (int i = 0; i < saida.length; i++) {
				bw.write(saida[i] + " ");
			}
			bw.newLine();

		} catch(Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Lê as informações da camada contida no arquivo.
	 * @param br leitor de buffer.
	 * @return instância de uma camada flatten.
	 */
	public Flatten lerConfig(BufferedReader br) {
		try {
			//formato de entrada
			String[] sEntrada = br.readLine().split(" ");
			int[] entrada = new int[sEntrada.length];
			for (int i = 0; i < sEntrada.length; i++) {
				entrada[i] = Integer.parseInt(sEntrada[i]);
			}

			//formato de saída
			String[] sSaida = br.readLine().split(" ");
			int[] saida = new int[sSaida.length];
			for (int i = 0; i < sSaida.length; i++) {
				saida[i] = Integer.parseInt(sSaida[i]);
			}

			Flatten camada = new Flatten(entrada);
			camada.construir(entrada);
			return camada;
		
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
}
