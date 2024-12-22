package jnn.serializacao;

import java.io.BufferedReader;

import jnn.camadas.Conv2D;
import jnn.core.tensor.Tensor;
import jnn.core.tensor.Variavel;

/**
 * Utilitário usado para serialização e desserialização de camadas Convolucionais.
 */
class SerialConv {

	public SerialConv() {}

	/**
	 * Transforma os dados contidos na camada Convolucional numa sequência
	 * de informações sequenciais. Essas informações contém:
	 * <ul>
	 *    <li> Nome da camada; </li>
	 *    <li> Formato de entrada (altura, largura, profundidade); </li>
	 *    <li> Formato de saída (altura, largura, profundidade); </li>
	 *    <li> Função de ativação configurada; </li>
	 *    <li> Uso de bias; </li>
	 *    <li> Valores dos filtros; </li>
	 *    <li> Valores dos bias (se houver); </li>
	 * </ul>
	 * @param camada camada convolucional que será serializada.
	 * @param sb StringBuilder usado como buffer.
	 * @param tipo tipo de dado que será escrito (double / float).
	 */
	public void serializar(Conv2D camada, StringBuilder sb, String tipo) {
		//nome da camada pra facilitar
		sb.append(camada.nome()).append("\n");

		//formato de entrada
		int[] entrada = camada.shapeEntrada();
		for (int i = 0; i < entrada.length; i++) {
			sb.append(entrada[i]).append(" ");
		}
		sb.append("\n");
		
		//formato de saída
		int[] saida = camada.shapeSaida();
		for (int i = 0; i < saida.length; i++) {
			sb.append(saida[i]).append(" ");
		}
		sb.append("\n");
		
		//formato dos filtros
		int[] shapeFiltro = camada.formatoFiltro();
		for (int i = 0; i < shapeFiltro.length; i++) {
			sb.append(shapeFiltro[i]).append(" ");
		}
		sb.append("\n");
		
		//função de ativação
		sb.append(camada.ativacao().nome()).append("\n");

		//bias
		sb.append(camada.temBias()).append("\n");

		//filtros
		Tensor filtros = camada.kernel();
		int[] shape = filtros.shape();
		for (int i = 0; i < shape[0]; i++) {
			for (int j = 0; j < shape[1]; j++) {
				for (int k = 0; k < shape[2]; k++) {
					for (int l = 0; l < shape[3]; l++) {
						escreverDado(filtros.get(i, j, k, l), tipo, sb);
						sb.append("\n");
					}
				}
			}
		}
		
		if(camada.temBias()){
			Variavel[] bias = camada.bias().paraArray();
			for(Variavel valor : bias){
				escreverDado(valor.get(), tipo, sb);
				sb.append("\n");               
			}
		}
	}

	/**
	 * Salva o valor de acordo com a configuração de tipo definida.
	 * @param valor valor desejado.
	 * @param tipo formatação do dado (float / double).
	 * @param sb StringBuilder usado como buffer..
	 */
	private void escreverDado(double valor, String tipo, StringBuilder sb) {
		tipo = tipo.toLowerCase();
		switch(tipo){
			case "float":
				sb.append((float) valor);
			break;

			case "double":
				sb.append(valor);
			break;
				
			default:
				throw new IllegalArgumentException("Tipo de dado (" + tipo + ") não suportado");
		}
	}

	/**
	 * Lê as informações da camada contida no arquivo.
	 * @param br leitor de buffer.
	 * @return instância de uma camada convolucional, os valores de
	 * filtros e bias ainda não são inicializados.
	 */
	public Conv2D lerConfig(BufferedReader br){
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

			//formato dos filtros
			String[] sFiltros = br.readLine().split(" ");
			int[] formFiltro = new int[sFiltros.length];
			for (int i = 0; i < sFiltros.length; i++) {
				formFiltro[i] = Integer.parseInt(sFiltros[i]);
			}
			
			//função de ativação
			String ativacao = br.readLine();

			//bias
			boolean bias = Boolean.valueOf(br.readLine());

			int numFiltros = saida[0];

			Conv2D camada = new Conv2D(numFiltros, formFiltro);
			camada.setAtivacao(ativacao);
			camada.setBias(bias);
			camada.construir(entrada);

			return camada;

		} catch (Exception e) {
			System.out.println("\nErro ao ler configurações da camada Convolucional:");
			throw new RuntimeException(e);
		}
	}

	/**
	 * Lê os valores dos filtros e bias para a camada.
	 * @param camada camada convolucional que será editada.
	 * @param br leitor de buffer.
	 */
	public void lerPesos(Conv2D camada, BufferedReader br){
		try {
			int tamKernel = camada.kernel().tam();
			Variavel[] arrKernel = new Variavel[tamKernel];

			for (int i = 0; i < tamKernel; i++) {
				arrKernel[i] = new Variavel(Double.parseDouble(br.readLine()));
			}

			camada.setKernel(arrKernel);
			
			if (camada.temBias()) {
				int tamBias = camada.bias().tam();
				Variavel[] arrBias = new Variavel[tamBias];

				for (int i = 0; i < tamBias; i++) {
					arrBias[i] = new Variavel(Double.parseDouble(br.readLine()));
				}
				
				camada.setBias(arrBias);
			}

		} catch (Exception e) {
			System.out.println("\nErro ao ler pesos da camada " + camada.nome());
			throw new RuntimeException(e);
		}
	}
}
