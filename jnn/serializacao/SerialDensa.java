package jnn.serializacao;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;

import jnn.camadas.Densa;
import jnn.core.tensor.Tensor;
import jnn.core.tensor.Variavel;

class SerialDensa {

	/**
	 * Transforma os dados contidos na camada Densa numa sequência
	 * de informações sequenciais. Essas informações contém:
	 * <ul>
	 *    <li> Nome da camada; </li>
	 *    <li> Formato de entrada (altura, largura); </li>
	 *    <li> Formato de saída (altura, largura); </li>
	 *    <li> Função de ativação configurada; </li>
	 *    <li> Uso de bias; </li>
	 *    <li> Valores dos pesos; </li>
	 *    <li> Valores dos bias (se houver); </li>
	 * </ul>
	 * @param camada camada densa que será serializada.
	 * @param bw escritor de buffer usado para salvar os dados da camada.
	 * @param tipo tipo de dado que será escrito.
	 */
	public void serializar(Densa camada, BufferedWriter bw, String tipo) {
		try {
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
			
			//função de ativação
			bw.write(String.valueOf(camada.ativacao().getClass().getSimpleName()));
			bw.newLine();

			//bias
			bw.write(String.valueOf(camada.temBias()));
			bw.newLine();
			
			Variavel[] pesos = camada.kernelParaArray();
			for (int i = 0; i < pesos.length; i++) {
				escreverDado(pesos[i].get(), tipo, bw);
				bw.newLine();
			}

			if (camada.temBias()) {
				Variavel[] bias = camada.biasParaArray();
				for (int i = 0; i < bias.length; i++) {
					escreverDado(bias[i].get(), tipo, bw);
					bw.newLine();
				}
			}
		
		} catch (IOException e) {
			System.out.println("\nErro ao serializar camada Densa:");
			e.printStackTrace();
		}
	}

	/**
	 * Salva o valor de acordo com a configuração de tipo definida.
	 * @param valor valor desejado.
	 * @param tipo formatação do dado (float, double).
	 * @param bw escritor de buffer usado.
	 */
	private void escreverDado(double valor, String tipo, BufferedWriter bw) throws IOException {
		tipo = tipo.toLowerCase();
		switch (tipo) {
			case "float":
				bw.write(String.valueOf((float) valor));
				break;

			case "double":
				bw.write(String.valueOf(valor));
				break;
				
			default:
				throw new IllegalArgumentException("Tipo de dado (" + tipo + ") não suportado");
		}
	}

	/**
	 * Lê as informações da camada contida no arquivo.
	 * @param br leitor de buffer.
	 * @return instância de uma camada densa, os valores de
	 * pesos e bias ainda não são inicializados.
	 */
	public Densa lerConfig(BufferedReader br) {
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
			
			//função de ativação
			String ativacao = br.readLine();

			//bias
			boolean bias = Boolean.valueOf(br.readLine());
			
			Densa camada = new Densa(saida[0], ativacao);
			camada.setBias(bias);
			camada.construir(entrada);
			return camada;

		} catch (Exception e) {
			System.out.println("\nErro ao ler configurações da camada Densa:");
			throw new RuntimeException(e);
		}
	}

	/**
	 * Lê os valores dos pesos e bias para a camada.
	 * @param camada camada densa que será editada.
	 * @param br leitor de buffer.
	 */
	public void lerPesos(Densa camada, BufferedReader br) {
		try {
			Tensor pesos = camada.kernel();
			int lin = pesos.shape()[0];
			int col = pesos.shape()[1];   
			double peso;     
			for(int i = 0; i < lin; i++){
				for(int j = 0; j < col; j++){
					peso = Double.parseDouble(br.readLine());
					camada.kernel().set(peso, i, j);
				}
			}
			
			if(camada.temBias()){
				Tensor bias = camada.bias();        
				int n = bias.tamanho();
				for(int i = 0; i < n; i++){
					double b = Double.parseDouble(br.readLine());
					bias.set(b, i);
				}
			}

		} catch (Exception e) {
			System.out.println("\nErro ao ler pesos da camada Densa:");
			throw new RuntimeException(e);
		}
	}
}
