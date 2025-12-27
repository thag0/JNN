package jnn.io;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import jnn.camadas.Camada;
import jnn.camadas.Conv2D;
import jnn.camadas.Densa;
import jnn.camadas.Dropout;
import jnn.camadas.Flatten;
import jnn.camadas.pooling.AvgPool2D;
import jnn.camadas.pooling.MaxPool2D;
import jnn.core.Dicionario;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerialCamada;
import jnn.metrica.perda.Perda;
import jnn.modelos.RedeNeural;
import jnn.modelos.Sequencial;
import jnn.otm.Otimizador;

/**
 * Classe responsável por tratar da gravação/leitura de modelos
 * de {@code RedeNeural} e {@code Sequencial}.
 */
public class Serializador extends SerialBase {

	/**
	 * Formato suportado de leitura e escrita dos modelos.
	 */
	private final String formatoModelo = ".nn";

	/**
	 * Operador de leitura/gravação.
	 */
	private SerialCamada serialCamada = new SerialCamada();

	/**
	 * Serializador e desserializador de modelos.
	 */
	public Serializador() {}

	/**
	 * Salva um modelo {@code RedeNeural} em um arquivo externo.
	 * @param modelo instância de uma {@code Rede Neural}.
	 * @param caminho caminho com nome e extensão do arquivo {@code .nn}.
	 * @param tipo tipo de valor usado na serialização, exemplo: {@code float} ou {@code double}.
	 */
	public void salvar(RedeNeural modelo, String caminho) {
		File arquivo = new File(caminho);
		if (!arquivo.getName().toLowerCase().endsWith(formatoModelo)) {
			throw new IllegalArgumentException(
				"\nO caminho especificado não é um arquivo de modelo válido."
			);
		}

		try (DataOutputStream out = new DataOutputStream(new FileOutputStream(arquivo))) {
			int[] arq = modelo.obterArquitetura();
			escrever(out, arq);
			escrever(out, modelo.temBias());
			escrever(out, modelo.otm().nome());
			escrever(out, modelo.loss().nome());

			for (Densa camada : modelo.camadas()) {
				serialCamada.serializar(camada, out);
			}

		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
	}

	/**
	 * Salva um modelo Sequencial em um arquivo externo.
	 * @param modelo modelo {@code Sequencial}.
	 * @param caminho caminho com nome e extensão do arquivo {@code .nn}.
	 * @param tipo tipo de valor usado na serialização, exemplo: {@code float} ou {@code double}.
	 */
	public void salvar(Sequencial modelo, String caminho) {
		File arquivo = new File(caminho);
		if (!arquivo.getName().toLowerCase().endsWith(formatoModelo)) {
			throw new IllegalArgumentException(
				"\nO caminho deve conter a extensão " + formatoModelo
			);
		}

		if (!modelo._compilado) {
			throw new IllegalStateException(
				"\nO modelo deve ser compilado."
			);
		}

		modelo.loteZero();

		try (DataOutputStream out = new DataOutputStream(new FileOutputStream(arquivo))) {
			escrever(out, modelo.numCamadas());
			escrever(out, modelo.otm().nome());
			escrever(out, modelo.loss().nome());

			for (Camada camada : modelo.camadas()) {
				serialCamada.serializar(camada, out);
			}

		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
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

        File arquivo = new File(caminho);
        if (!arquivo.getName().toLowerCase().endsWith(formatoModelo)) {
            throw new IllegalArgumentException("O caminho deve conter a extensão " + formatoModelo);
        }

		try (DataInputStream in = new DataInputStream(new FileInputStream(arquivo))) {
			int[] arq = lerArrInt(in);
			boolean temBias = lerBoolean(in);
			Otimizador otm = dicio.getOtimizador(lerString(in));
			Perda loss = dicio.getPerda(lerString(in));

			rede = new RedeNeural(arq);
			rede.configurarBias(temBias);
			rede.compilar(otm, loss);

			int numCamadas = rede.numCamadas();
			Densa[] camadas = new Densa[numCamadas];

			for (int i = 0; i < numCamadas; i++) {
				// ler o nome da camada porque a api que escreve
				// a densa faz isso no começo.
				// mesmo sabendo que só tem densa na mlp, não da pra ignorar
				String nome = lerString(in);
				camadas[i] = (Densa) serialCamada.ler(in, nome);
			}

			for (int i = 0; i < numCamadas; i++) {
				rede.camada(i).kernel().copiar(camadas[i].kernel());
				if (temBias) rede.camada(i).bias().copiar(camadas[i].bias());

				rede.camada(i).setAtivacao(camadas[i].ativacao().nome());
			}

		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
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

        File arquivo = new File(caminho);
        if (!arquivo.getName().toLowerCase().endsWith(formatoModelo)) {
            throw new IllegalArgumentException("O caminho deve conter a extensão " + formatoModelo);
        }

		try (DataInputStream in = new DataInputStream(new FileInputStream(arquivo))) {
			int numCamadas = lerInt(in);
			String otmStr = lerString(in);
			String lossStr = lerString(in);

			for (int i = 0; i < numCamadas; i++) {
				String nomeCamada = lerString(in).toLowerCase();

				switch (nomeCamada) {
					case "densa":
						Densa densa = (Densa) serialCamada.ler(in, nomeCamada);
						modelo.add(densa);
					break;

					case "conv2d":
						Conv2D conv2d = (Conv2D) serialCamada.ler(in, nomeCamada);
						modelo.add(conv2d);
					break;

					case "flatten":
						Flatten flatten = (Flatten) serialCamada.ler(in, nomeCamada);
						modelo.add(flatten);
					break;

					case "maxpool2d":
						MaxPool2D maxpool = (MaxPool2D) serialCamada.ler(in, nomeCamada);
						modelo.add(maxpool);
					break;

					case "avgpool2d":
						AvgPool2D avgpool = (AvgPool2D) serialCamada.ler(in, nomeCamada);
						modelo.add(avgpool);
					break;

					case "dropout":
						Dropout dropout = (Dropout) serialCamada.ler(in, nomeCamada);
						modelo.add(dropout);
					break;
				
					default:
						throw new UnsupportedOperationException(
							"\nCamada " + nomeCamada + " não suportada."
						);
				}
			}

			modelo._compilado = true;
			for (int i = 0; i < modelo.numCamadas(); i++) {
				modelo.camada(i).setId(i);
			}

			Otimizador otm = dicio.getOtimizador(dicio.getOtimizador(otmStr));
			Perda loss = dicio.getPerda(dicio.getPerda(lossStr));

			modelo.setOtimizador(otm);
			otm.construir(modelo.params(), modelo.grads());
			
			modelo.setPerda(loss);

		} catch (IOException e) {
			System.out.println("Erro ao ler o modelo.");
			e.printStackTrace();
			System.exit(1);
		}

		return modelo;
	}
}
