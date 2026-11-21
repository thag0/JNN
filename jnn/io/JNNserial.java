package jnn.io;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import jnn.core.JNNdict;
import jnn.core.JNNutils;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.camadas.LayerSerializer;
import jnn.layers.BatchNorm2D;
import jnn.layers.Conv2D;
import jnn.layers.Dense;
import jnn.layers.Dropout;
import jnn.layers.Flatten;
import jnn.layers.Layer;
import jnn.layers.acts.ELU;
import jnn.layers.acts.GELU;
import jnn.layers.acts.LeakyReLU;
import jnn.layers.acts.ReLU;
import jnn.layers.acts.SELU;
import jnn.layers.acts.Sigmoid;
import jnn.layers.acts.Softmax;
import jnn.layers.acts.Softplus;
import jnn.layers.acts.Swish;
import jnn.layers.acts.Tanh;
import jnn.layers.pooling.AvgPool2D;
import jnn.layers.pooling.GlobalAvgPool2D;
import jnn.layers.pooling.MaxPool2D;
import jnn.models.Sequential;

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
	static private LayerSerializer serializer = new LayerSerializer();

	/**
	 * Serializador e desserializador de modelos.
	 */
	private JNNserial() {}

	/**
	 * Salva um modelo Sequencial em um arquivo externo.
	 * @param modelo modelo {@code Sequencial}.
	 * @param caminho caminho com nome e extensão do arquivo {@code .nn}.
	 */
	static public void salvar(Sequential modelo, String caminho) {
		File arquivo = new File(caminho);
		if (!arquivo.getName().toLowerCase().endsWith(formatoModelo)) {
			throw new IllegalArgumentException(
				"\nO caminho deve conter a extensão " + formatoModelo
			);
		}

		// garantia de compilação do modelo e
		// de que as dimensões das camadas estão
		modelo.zeroBatch();

		try (DataOutputStream out = new DataOutputStream(new FileOutputStream(arquivo))) {
			write(out, modelo.numLayers());
			write(out, modelo.optim().nome());
			write(out, modelo.loss().name());

			for (Layer camada : modelo.layers()) {
				serializer.serialize(camada, out);
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
	static public Sequential lerSequencial(String caminho) {
		File arquivo = new File(caminho);
        if (!arquivo.getName().toLowerCase().endsWith(formatoModelo)) {
			throw new IllegalArgumentException("O caminho deve conter a extensão " + formatoModelo);
        }
		
		Layer[] layers = {};
		String strOptim = "";
		String strLoss 	= "";
		
		try (DataInputStream in = new DataInputStream(new FileInputStream(arquivo))) {
			int numCamadas = readInt(in);
			strOptim = readString(in);
			strLoss = readString(in);

			for (int i = 0; i < numCamadas; i++) {
				String nomeCamada = readString(in).toLowerCase();

				switch (nomeCamada) {
					case "dense":
						layers = JNNutils.addToArray(layers, (Dense) serializer.read(in, nomeCamada));
					break;

					case "conv2d":
						layers = JNNutils.addToArray(layers, (Conv2D) serializer.read(in, nomeCamada));
					break;

					case "batchnorm2d":
						layers = JNNutils.addToArray(layers, (BatchNorm2D) serializer.read(in, nomeCamada));
					break;

					case "flatten":
						layers = JNNutils.addToArray(layers, (Flatten) serializer.read(in, nomeCamada));
					break;

					case "maxpool2d":
						layers = JNNutils.addToArray(layers, (MaxPool2D) serializer.read(in, nomeCamada));
					break;

					case "avgpool2d":
						layers = JNNutils.addToArray(layers, (AvgPool2D) serializer.read(in, nomeCamada));
					break;

					case "globalavgpool2d":
						layers = JNNutils.addToArray(layers, (GlobalAvgPool2D) serializer.read(in, nomeCamada));
					break;

					case "dropout":
						layers = JNNutils.addToArray(layers, (Dropout) serializer.read(in, nomeCamada));
					break;

					case "elu":
						layers = JNNutils.addToArray(layers, (ELU) serializer.read(in, nomeCamada));
					break;
					
					case "gelu":
						layers = JNNutils.addToArray(layers, (GELU) serializer.read(in, nomeCamada));
					break;

					case "leakyrelu":
						layers = JNNutils.addToArray(layers, (LeakyReLU) serializer.read(in, nomeCamada));
					break;

					case "relu":
						layers = JNNutils.addToArray(layers, (ReLU) serializer.read(in, nomeCamada));
					break;

					case "selu":
						layers = JNNutils.addToArray(layers, (SELU) serializer.read(in, nomeCamada));
					break;

					case "sigmoid":
						layers = JNNutils.addToArray(layers, (Sigmoid) serializer.read(in, nomeCamada));
					break;
						
					case "softmax":
						layers = JNNutils.addToArray(layers, (Softmax) serializer.read(in, nomeCamada));
					break;

					case "softplus":
						layers = JNNutils.addToArray(layers, (Softplus) serializer.read(in, nomeCamada));
					break;

					case "swish":
						layers = JNNutils.addToArray(layers, (Swish) serializer.read(in, nomeCamada));
					break;

					case "tanh":
						layers = JNNutils.addToArray(layers, (Tanh) serializer.read(in, nomeCamada));
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

		Sequential model = new Sequential(layers);
		
		// mudar isso depois para algo como "modelo.compilar(otm, loss, false)"
		// onde o false faria com que os parametros lidos não fossem mexidos na compilação
		model._compiled = true;

		for (int i = 0; i < model.numLayers(); i++) {
			model.layer(i).setId(i);
		}

		model.setOptimizer(JNNdict.getOptim(strOptim));
		model.optim().construir(model.params());
		model.setLoss(JNNdict.getLoss(strLoss));

		return model;
	}
}
