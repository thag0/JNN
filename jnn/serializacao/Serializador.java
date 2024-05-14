package jnn.serializacao;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

import jnn.avaliacao.perda.Perda;
import jnn.camadas.AvgPool2D;
import jnn.camadas.Camada;
import jnn.camadas.Conv2D;
import jnn.camadas.Densa;
import jnn.camadas.Dropout;
import jnn.camadas.Flatten;
import jnn.camadas.MaxPool2D;
import jnn.core.Dicionario;
import jnn.modelos.RedeNeural;
import jnn.modelos.Sequencial;
import jnn.otimizadores.Otimizador;

/**
 * Classe responsável por tratar da serialização/desserialização de modelos
 * de {@Rede Neural} e {@code Sequencial}.
 * <p>
 *    Manipula os arquivos {@code .txt} baseados na rede para escrita e leitura, 
 *    possibilitando mais portabilidade de modelos treinados.
 * </p>
 */
public class Serializador {

	/**
	 * Formato suportado de leitura e escrita dos modelos.
	 */
	private final String formatoModelo = ".nn";

	/**
	 * Auxiliar na serialização de camadas densas.
	 */
	private SerialDensa auxDensa = new SerialDensa();
	
	/**
	 * Auxiliar na serialização de camadas convolucionais.
	 */
	private SerialConv auxConv = new SerialConv();

	/**
	 * Auxiliar na serialização de camadas flatten.
	 */
	private SerialFlatten auxFlat = new SerialFlatten();

	/**
	 * Auxiliar na serialização de camadas max pooling.
	 */
	private SerialMaxPool auxMaxPool = new SerialMaxPool();

	/**
	 * Auxiliar na serialização de camadas avg pooling.
	 */
	private SerialAvgPool auxAvgPool = new SerialAvgPool();

	/**
	 * Auxiliar na serialização de camadas de dropout.
	 */
	private SerialDropout auxDropout = new SerialDropout();

	/**
	 * Serializador e desserializador de modelos.
	 */
	public Serializador() {}

	/**
	 * Salva um modelo Sequencial em um arquivo externo.
	 * <p>
	 *    Os valores salvos estarão no formato {@code double}.
	 * </p>
	 * @param rede instância de uma {@code Rede Neural}.
	 * @param caminho caminho com nome e extensão do arquivo {@code .nn}.
	 */
	public void salvar(RedeNeural rede, String caminho) {
		salvar(rede, caminho, "double");
	}

	/**
	 * Salva um modelo Sequencial em um arquivo externo.
	 * @param rede instância de uma {@code Rede Neural}.
	 * @param caminho caminho com nome e extensão do arquivo {@code .nn}.
	 * @param tipo tipo de valor usado na serialização, exemplo: {@code float} ou {@code double}.
	 */
	public void salvar(RedeNeural rede, String caminho, String tipo) {
		File arquivo = new File(caminho);
		if (!arquivo.getName().toLowerCase().endsWith(formatoModelo)) {
			throw new IllegalArgumentException(
				"\nO caminho especificado não é um arquivo de modelo válido."
			);
		}

		try (BufferedWriter bw = new BufferedWriter(new FileWriter(arquivo))) {
			//arquitetura da rede
			int[] arq = rede.obterArquitetura();
			for (int i = 0; i < arq.length; i++) {
				bw.write(arq[i] + " ");
			}
			bw.newLine();

			//bias
			bw.write(Boolean.toString(rede.temBias()));
			bw.newLine();

			//funções de ativação
			Densa[] camadas = rede.camadas();
			for (int i = 0; i < camadas.length; i++) {
				bw.write(camadas[i].ativacao().getClass().getSimpleName());
				bw.write(" ");
			}
			bw.newLine();

			//pesos dos neuronios
			for (Densa camada : rede.camadas()) {
				auxDensa.serializar(camada, bw, tipo);
			}

		} catch(Exception e) {
			System.out.println("\nErro ao salvar o arquivo da Rede Neural.");
			e.printStackTrace();
		}
	}

	/**
	 * Salva um modelo Sequencial em um arquivo externo.
	 * <p>
	 *    Os valores salvos estarão no formato {@code double}.
	 * </p>
	 * @param modelo modelo {@code Sequencial}.
	 * @param caminho caminho com nome e extensão do arquivo {@code .nn}.
	 */
	public void salvar(Sequencial modelo, String caminho) {
		salvar(modelo, caminho, "double");
	}

	/**
	 * Salva um modelo Sequencial em um arquivo externo.
	 * @param modelo modelo {@code Sequencial}.
	 * @param caminho caminho com nome e extensão do arquivo {@code .nn}.
	 * @param tipo tipo de valor usado na serialização, exemplo: {@code float} ou {@code double}.
	 */
	public void salvar(Sequencial modelo, String caminho, String tipo) {
		File arquivo = new File(caminho);
		if (!arquivo.getName().toLowerCase().endsWith(formatoModelo)) {
			throw new IllegalArgumentException(
				"\nO caminho deve conter a extensão " + formatoModelo
			);
		}

		try (BufferedWriter bw = new BufferedWriter(new FileWriter(arquivo))) {
			//quantidade de camadas
			bw.write(String.valueOf(modelo.numCamadas()));
			bw.newLine();

			//otimizador usado
			bw.write(modelo.otimizador().nome());
			bw.newLine();

			//função de perda
			bw.write(modelo.perda().nome());
			bw.newLine();

			for (Camada camada : modelo.camadas()) {
				if (camada instanceof Densa) {
					auxDensa.serializar((Densa) camada, bw, tipo);

				} else if (camada instanceof Conv2D) {
					auxConv.serializar((Conv2D) camada, bw, tipo);
				
				} else if (camada instanceof Flatten) {
					auxFlat.serializar((Flatten) camada, bw);
				
				} else if (camada instanceof MaxPool2D) {
					auxMaxPool.serializar((MaxPool2D) camada, bw);

				} else if (camada instanceof AvgPool2D) {
					auxAvgPool.serializar((AvgPool2D) camada, bw);

				} else if (camada instanceof Dropout) {
					auxDropout.serializar((Dropout) camada, bw);

				} else{
					throw new IllegalArgumentException(
						"Tipo de camada \"" + camada.getClass().getTypeName() + "\" não suportado."
					);
				}
			}
		
		} catch(Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Lê o arquivo de uma {@code Rede Neural} serializada e converte numa
	 * instância pré configurada.
	 * @param caminho caminho onde está salvo o arquivo {@code .nn} do modelo.
	 * @return modelo {@code RedeNeural} baseado no arquivo lido.
	 */
	public RedeNeural lerRedeNeural(String caminho) {
		RedeNeural rede = null;
		Dicionario dicio = new Dicionario();

		try (BufferedReader br = new BufferedReader(new FileReader(caminho))) {
			//arquitetura
			String[] arqStr = br.readLine().split(" ");
			int[] arq = new int[arqStr.length];

			try {
				for (int i = 0; i < arqStr.length; i++) {
					arq[i] = Integer.parseInt(arqStr[i]);
				}
			} catch(Exception e) {
				System.out.println("\nOcorreu um erro ao tentar ler os valores de arquitetura");
				System.out.println("Verifique se estão corretamente formatados");
				System.out.println("Cada elemento de arquitetura deve ser separado por espaços");
				System.out.println("Ex: \"2 3 4\"");
				System.exit(0);
			}

			//bias
			boolean bias = Boolean.parseBoolean(br.readLine());

			//funções de ativação
			String[] ativacoesStr = br.readLine().split(" ");

			//inicialização e configurações da rede
			rede = new RedeNeural(arq);
			rede.configurarBias(bias);
			rede.compilar();

			for (int i = 0; i < rede.numCamadas(); i++) {
				rede.configurarAtivacao(rede.camada(i), dicio.getAtivacao(ativacoesStr[i]));
			}

			for (int i = 0; i < rede.numCamadas(); i++) {
				String nome = br.readLine();
				if (nome.equals("Densa")) {
					br.readLine();//entrada
					br.readLine();//saida
					rede.camada(i).setAtivacao(br.readLine());
					rede.configurarBias(Boolean.valueOf(br.readLine()));
					auxDensa.lerPesos(rede.camada(i), br);
				}
			}

		} catch (Exception e) {
			System.out.println("\nErro ao ler o arquivo de Rede Neural \""+ caminho + "\".");
			e.printStackTrace();
			System.exit(0);
		}

		return rede;
	}

	/**
	 * Lê o arquivo de um modelo {@code Sequencial} serializado e converte numa
	 * instância pré configurada.
	 * @param caminho caminho onde está saldo o arquivo {@code .nn} do modelo;
	 * @return modelo {@code Sequencial} lido a partir do arquivo.
	 */
	public Sequencial lerSequencial(String caminho) {
		Sequencial modelo = new Sequencial();
		Dicionario dicio = new Dicionario();

		try (BufferedReader br = new BufferedReader(new FileReader(caminho))) {
			int numCamadas = Integer.parseInt(br.readLine());
			Otimizador otimizador = dicio.getOtimizador(br.readLine().trim());
			Perda perda = dicio.getPerda(br.readLine().trim());
		
			modelo.setOtimizador(otimizador);
			modelo.setPerda(perda);
			for (int i = 0; i < numCamadas; i++) {
				String nome = br.readLine();
				
				if (nome.equalsIgnoreCase("densa")) {
					Densa densa = auxDensa.lerConfig(br);
					auxDensa.lerPesos(densa, br);
					modelo.add(densa);

				} else if (nome.equalsIgnoreCase("conv2d")) {
					Conv2D convolucional = auxConv.lerConfig(br);
					auxConv.lerPesos(convolucional, br);
					modelo.add(convolucional);
				
				} else if (nome.equalsIgnoreCase("flatten")) {
					Flatten flat = auxFlat.lerConfig(br);
					modelo.add(flat);
				
				} else if (nome.equalsIgnoreCase("maxpool2d")) {
					MaxPool2D maxPooling = auxMaxPool.lerConfig(br);
					modelo.add(maxPooling);
				
				} else if (nome.equalsIgnoreCase("avgpool2d")) {
					AvgPool2D avgPooling = auxAvgPool.lerConfig(br);
					modelo.add(avgPooling);

				} else if (nome.equalsIgnoreCase("dropout")) {
					Dropout dropout = auxDropout.lerConfig(br);
					modelo.add(dropout);

				} else {
					throw new IllegalArgumentException(
						"Tipo de camada \""+ nome +"\" não suportado."
					);
				}
			}

			modelo._compilado = true;
			for (int i = 0; i < modelo.numCamadas(); i++) {
				modelo.camada(i).setId(i);
			}
			otimizador.construir(modelo.camadas());

		} catch(Exception e) {
			throw new RuntimeException(e);
		}

		return modelo;
	}
}
