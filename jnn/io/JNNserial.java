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
import jnn.camadas.acts.ELU;
import jnn.camadas.acts.GELU;
import jnn.camadas.acts.LeakyReLU;
import jnn.camadas.acts.ReLU;
import jnn.camadas.acts.SELU;
import jnn.camadas.acts.Sigmoid;
import jnn.camadas.acts.Softmax;
import jnn.camadas.acts.Softplus;
import jnn.camadas.acts.Swish;
import jnn.camadas.acts.Tanh;
import jnn.camadas.pooling.AvgPool2D;
import jnn.camadas.pooling.GlobalAvgPool2D;
import jnn.camadas.pooling.MaxPool2D;
import jnn.core.Dicionario;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.camadas.SerialCamada;
import jnn.metrica.perda.Perda;
import jnn.modelos.Sequencial;
import jnn.otm.Otimizador;

/**
 * Classe responsável por tratar da gravação/leitura de modelos
 * de {@code RedeNeural} e {@code Sequencial}.
 */
public class JNNserial extends SerialBase {

	/**
	 * Formato suportado de leitura e escrita dos modelos.
	 */
	static private final String formatoModelo = ".nn";

	/**
	 * Operador de leitura/gravação.
	 */
	static private SerialCamada serialCamada = new SerialCamada();

	/**
	 * Serializador e desserializador de modelos.
	 */
	private JNNserial() {}

	/**
	 * Salva um modelo Sequencial em um arquivo externo.
	 * @param modelo modelo {@code Sequencial}.
	 * @param caminho caminho com nome e extensão do arquivo {@code .nn}.
	 */
	static public void salvar(Sequencial modelo, String caminho) {
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
	 * Lê o arquivo de um modelo {@code Sequencial} serializado e converte numa
	 * instância pré configurada.
	 * @param caminho caminho onde está saldo o arquivo {@code .nn} do modelo;
	 * @return modelo {@code Sequencial} lido a partir do arquivo.
	 */
	static public Sequencial lerSequencial(String caminho) {
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
						modelo.add((Densa) serialCamada.ler(in, nomeCamada));
					break;

					case "conv2d":
						modelo.add((Conv2D) serialCamada.ler(in, nomeCamada));
					break;

					case "flatten":
						modelo.add((Flatten) serialCamada.ler(in, nomeCamada));
					break;

					case "maxpool2d":
						modelo.add((MaxPool2D) serialCamada.ler(in, nomeCamada));
					break;

					case "avgpool2d":
						modelo.add((AvgPool2D) serialCamada.ler(in, nomeCamada));
					break;

					case "gap":
						modelo.add((GlobalAvgPool2D) serialCamada.ler(in, nomeCamada));
					break;

					case "dropout":
						modelo.add((Dropout) serialCamada.ler(in, nomeCamada));
					break;

					case "elu":
						modelo.add((ELU) serialCamada.ler(in, nomeCamada));
					break;
					
					case "gelu":
						modelo.add((GELU) serialCamada.ler(in, nomeCamada));
					break;

					case "leakyrelu":
						modelo.add((LeakyReLU) serialCamada.ler(in, nomeCamada));
					break;

					case "relu":
						modelo.add((ReLU) serialCamada.ler(in, nomeCamada));
					break;

					case "selu":
						modelo.add((SELU) serialCamada.ler(in, nomeCamada));
					break;

					case "sigmoid":
						modelo.add((Sigmoid) serialCamada.ler(in, nomeCamada));
					break;

					case "softmax":
						modelo.add((Softmax) serialCamada.ler(in, nomeCamada));
					break;

					case "softplus":
						modelo.add((Softplus) serialCamada.ler(in, nomeCamada));
					break;

					case "swish":
						modelo.add((Swish) serialCamada.ler(in, nomeCamada));
					break;

					case "tanh":
						modelo.add((Tanh) serialCamada.ler(in, nomeCamada));
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
