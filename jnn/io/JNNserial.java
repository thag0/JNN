package jnn.io;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import jnn.camadas.BatchNorm2D;
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
import jnn.core.JNNutils;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.camadas.SerialCamada;
import jnn.modelos.Sequencial;

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

		// garantia de compilação do modelo e
		// de que as dimensões das camadas estão
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
		File arquivo = new File(caminho);
        if (!arquivo.getName().toLowerCase().endsWith(formatoModelo)) {
			throw new IllegalArgumentException("O caminho deve conter a extensão " + formatoModelo);
        }
		
		Dicionario dicio = new Dicionario();
		Camada[] cs = {};
		String otmStr = "";
		String lossStr = "";
		
		try (DataInputStream in = new DataInputStream(new FileInputStream(arquivo))) {
			int numCamadas = lerInt(in);
			otmStr = lerString(in);
			lossStr = lerString(in);

			for (int i = 0; i < numCamadas; i++) {
				String nomeCamada = lerString(in).toLowerCase();

				switch (nomeCamada) {
					case "densa":
						cs = JNNutils.addEmArray(cs, (Densa) serialCamada.ler(in, nomeCamada));
					break;

					case "conv2d":
						cs = JNNutils.addEmArray(cs, (Conv2D) serialCamada.ler(in, nomeCamada));
					break;

					case "batchnorm2d":
						cs = JNNutils.addEmArray(cs, (BatchNorm2D) serialCamada.ler(in, nomeCamada));
					break;

					case "flatten":
						cs = JNNutils.addEmArray(cs, (Flatten) serialCamada.ler(in, nomeCamada));
					break;

					case "maxpool2d":
						cs = JNNutils.addEmArray(cs, (MaxPool2D) serialCamada.ler(in, nomeCamada));
					break;

					case "avgpool2d":
						cs = JNNutils.addEmArray(cs, (AvgPool2D) serialCamada.ler(in, nomeCamada));
					break;

					case "globalavgpool2d":
						cs = JNNutils.addEmArray(cs, (GlobalAvgPool2D) serialCamada.ler(in, nomeCamada));
					break;

					case "dropout":
						cs = JNNutils.addEmArray(cs, (Dropout) serialCamada.ler(in, nomeCamada));
					break;

					case "elu":
						cs = JNNutils.addEmArray(cs, (ELU) serialCamada.ler(in, nomeCamada));
					break;
					
					case "gelu":
						cs = JNNutils.addEmArray(cs, (GELU) serialCamada.ler(in, nomeCamada));
					break;

					case "leakyrelu":
						cs = JNNutils.addEmArray(cs, (LeakyReLU) serialCamada.ler(in, nomeCamada));
					break;

					case "relu":
						cs = JNNutils.addEmArray(cs, (ReLU) serialCamada.ler(in, nomeCamada));
					break;

					case "selu":
						cs = JNNutils.addEmArray(cs, (SELU) serialCamada.ler(in, nomeCamada));
					break;

					case "sigmoid":
						cs = JNNutils.addEmArray(cs, (Sigmoid) serialCamada.ler(in, nomeCamada));
					break;
						
					case "softmax":
						cs = JNNutils.addEmArray(cs, (Softmax) serialCamada.ler(in, nomeCamada));
					break;

					case "softplus":
						cs = JNNutils.addEmArray(cs, (Softplus) serialCamada.ler(in, nomeCamada));
					break;

					case "swish":
						cs = JNNutils.addEmArray(cs, (Swish) serialCamada.ler(in, nomeCamada));
					break;

					case "tanh":
						cs = JNNutils.addEmArray(cs, (Tanh) serialCamada.ler(in, nomeCamada));
					break;
				
					default:
						throw new UnsupportedOperationException(
							"\nCamada " + nomeCamada + " não suportada."
						);
				}
			}
		} catch (IOException e) {
			System.out.println("Erro ao ler o modelo.");
			e.printStackTrace();
			System.exit(1);
		}

		Sequencial modelo = new Sequencial(cs);
		for (int i = 0; i < modelo.numCamadas(); i++) modelo.camada(i).setId(i);
		modelo.setOtimizador(dicio.getOtimizador(otmStr));
		modelo.otm().construir(modelo.params(), modelo.grads());
		modelo.setPerda(dicio.getPerda(lossStr));

		// mudar isso depois para algo como "modelo.compilar(otm, loss, false)"
		// onde o false faria com que os parametros lidos não fossem mexidos na compilação
		modelo._compilado = true;

		return modelo;
	}
}
