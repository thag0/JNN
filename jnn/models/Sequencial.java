package jnn.models;

import java.util.Iterator;

import jnn.core.Dicionario;
import jnn.core.JNNutils;
import jnn.layers.Input;
import jnn.layers.Layer;
import jnn.metrics.Evaluator;
import jnn.treino.Treinador;

/**
 * <p>
 *    Modelo sequencial de camadas
 * </p>
 * <p>
 *    Uma interface simples para criação de modelos de redes neurais,
 *    empilhando camadas em sequência que podem ser customizáveis.
 * </p>
 * <h2>
 *    Criação
 * </h2>
 * <p>
 *    Para qualquer modelo novo, é sempre necessário informar o formato
 *    de entrada da primeira camada contida nele. Caso não seja de interesse,
 *    é possível adicionar a camada {@code Entrada} que contém o formato desejado.
 * </p>
 * <p>
 *    Exemplos:
 * </p>
 * <pre>
 *modelo = Sequencial();
 *modelo.add(new Densa(2, 3));
 *modelo.add(new Densa(2));
 *
 *modelo = Sequencial(
 *    new Densa(2, 3),
 *    new Densa(2)
 *);
 *
 *modelo = Sequencial(
 *    new Entrada(2),
 *    new Densa(3),
 *    new Densa(2)
 *);
 * </pre>
 * O modelo sequencial não é limitado apenas a camadas densas, podendo empilhar camadas
 * compatívels que herdam de {@code jnn.camadas.Camada}, algumas camadas dispoíveis incluem:
 * <ul>
 *    <li> Entrada; </li>
 *    <li> Densa; </li>
 *    <li> Conv2D; </li>
 *    <li> MaxPool2D; </li>
 *    <li> AvgPool2D; </li>
 *    <li> Flatten; </li>
 *    <li> Dropout; </li>
 * </ul>
 * <p>
 *    Exemplo:
 * </p>
 * <pre>
 *modelo = Sequencial(
 *    new Entrada(28, 28),
 *    new Conv2D(5, new int[]{3, 3}),
 *    new MaxPool2D(new int[]{2, 2}),
 *    new Flatten(),
 *    new Densa(50),
 *    new Dropout(0.3),
 *    new Densa(10),
 *);
 * </pre>
 * <h2>
 *    Compilação
 * </h2>
 * <p>
 *    Para poder usar o modelo é necessário compilá-lo, informando qual otimizador
 *    e função de perda serão utilizados pelo modelo.
 * </p>
 *    Exemplo:
 * <pre>
 *modelo.compilar("sgd", "mse");
 *modelo.compilar(new SGD(0.01, 0.9), new MSE());
 * </pre>
 * <h2>
 *    Predições
 * </h2>
 * <p>
 *    Obter os resultados previstos pelo medelo pode ser facilmente feito apenas o alimentando
 *    com dados de entrada. Os dados dados de entrada para maior facilidade de manipulação podem
 *    ser instâncias de um {@code Tensor} (tanto únicas, quanto arrays), mas o modelo não é
 *    limitado a isso, sendo necesário apenas saber se a camada inicial possui suporte para o
 *    dado fornecido.
 * </p>
 * <p>
 *    Exemplo:
 * </p>
 * <pre>
 *Tensor entrada = ...;
 *Tensor pred = modelo.forward(entrada);//Obtendo uma única predição
 *
 *Tensor[] entradas = ...;
 *Tensor[] preds = modelo.forwards(entrada);//Obtendo várias predições
 *</pre>
 * <h2>
 *    Treinamento
 * </h2>
 * <p>
 *    Modelos sequenciais podem ser facilmente treinados usando o método {@code treinar},
 *    onde é apenas necessário informar os dados de entrada, saída e a quantidade de épocas 
 *    desejada para treinar.
 * </p>
 * Exemplo:
 * <pre>
 *Object[] treinoX = ...; //dados de entrada
 *Object[] treinoY = ...; //dados de saída
 *int epochs = ... ; //iterações dentro do conjunto de dados
 *boolean logs = ...; //impressão de perda do modelo durante o treino.
 *modelo.treinar(treinoX, treinoY, epochs, logs);
 * </pre>
 * <h2>
 *    Serialização
 * </h2>
 * <p>
 *    Modelos sequenciais podem ser salvos em arquivos externos {@code .txt} para preservar
 *    suas configurações mais importantes, como otimizador, função de perda e mais importante
 *    ainda, as configurações de cada camada, isso inclue os valores para os kernels e bias
 *    contidos em cada camada treinável além dos formatos para entrada e saída específicos.
 * </p>
 * <p>
 *    Para salvar o modelo deve-se fazer uso da classe Serializador disponível em {@code rna.serializacao.Serializador}
 * </p>
 * Exemplo:
 * <pre>
 *Sequencial modelo = //modelo já configurado e compilado
 *Serializador s = new Serializador();
 *String caminho = "./modelo.txt";
 *s.salvar(modelo, caminho);
 * </pre>
 * <h2>
 *    Desserialização
 * </h2>
 * <p>
 *    Como esperado após salvar, também é possível ler um modelo sequencial a partir de um
 *    arquivo gerado pelo serializador. Esse arquivo deve ser compatível com as configurações
 *    salvas pelo serializador.
 * </p>
 * Exemplo:
 * <pre>
 *Serializador s = new Serializador();
 *String caminho = "./modelo.txt";
 *Sequencial modelo = s.lerSequencial(caminho);
 * </pre>
 * @author Thiago Barroso, acadêmico de Engenharia da Computação pela Universidade Federal do Pará, 
 * Campus Tucuruí. Dezembro/2023.
 */
public class Sequencial extends Model {

	/**
	 * Lista de camadas do modelo.
	 */
	private Layer[] _camadas;

	/**
	 * Instancia um modelo sequencial com o conjunto de camadas vazio.
	 * <p>
	 *    É necessário especificar o formato de entrada da primeira camada
	 *    do modelo.
	 * </p>
	 * <p>
	 *    As camadas do modelo deverão ser adicionadas manualmente
	 *    usando o método {@code add()}.
	 * </p>
	 */
	public Sequencial() {
		_camadas = new Layer[0];
		_compiled = false;
	}

	/**
	 * Inicializa um modelo sequencial a partir de um conjunto de camadas
	 * definido
	 * <p>
	 *    É necessário especificar o formato de entrada da primeira camada
	 *    do modelo.
	 * </p>
	 * @param camadas conjunto de camadas que serão usadas pelo modelo.
	 * @throws IllegalArgumentException caso o conjunto de camadas seja nulo
	 * ou alguma camada contida seja.
	 */
	public Sequencial(Layer... camadas) {
		this();// evitar repetição de código

		JNNutils.validarNaoNulo(camadas, "camadas == null.");

		if (camadas.length < 1) {
			throw new IllegalArgumentException(
				"\nConjunto de camadas vazio."
			);
		}
		
		add(camadas[0]);
		for (int i = 1; i < camadas.length; i++) {
			if (!(camadas[i] instanceof Input)) add(camadas[i]);
		}
	}

	/**
	 * Adiciona uma nova camada ao final da lista de camadas do modelo.
	 * <p>
	 *    Novas camadas não precisam estar construídas, a única excessão
	 *    é caso seja a primeira camada do modelo, ela deve ser construída
	 *    já que é necessário saber o formato de entrada do modelo.
	 * </p>
	 * Ao adicionar novas camadas, o modelo precisará ser compilado novamente.
	 * @param c nova camada.
	 * @throws IllegalArgumentException se a camada fornecida for nula,
	 */
	public void add(Layer c) {
		_camadas = JNNutils.addEmArray(_camadas, c);
		_compiled = false;
	}

	/**
	 * Remove a última camada contida na lista de camadas do modelo.
	 * @return camada removida.
	 * @throws IllegalArgumentException caso o modelo já não possua nenhuma 
	 * camada disponível.
	 */
	public Layer rem() {
		if (_camadas.length < 1) {
			throw new IllegalArgumentException(
				"\nNão há camadas no modelo."
			);
		}

		Layer ultima = outputLayer();

		Layer[] novas = _camadas;
		_camadas = new Layer[_camadas.length-1];
		System.arraycopy(novas, 0, _camadas, 0, _camadas.length);

		_compiled = false;

		return ultima;
	}

	@Override
	public void compile(Object otm, Object loss) {
		if (layers().length < 1) {
			throw new IllegalStateException(
				"\nNão há camadas no modelo."
			);
		}

		if (layer(0) instanceof Input) {
			int[] shape = layer(0).shapeOut();

			// remover camada de entrada do modelo
			Layer[] temp = layers();
			_camadas = new Layer[numLayers()-1];
			System.arraycopy(temp, 1, _camadas, 0, numLayers());

			if (numLayers() < 1) {
				throw new IllegalStateException(
					"\nO modelo não possui camadas para compilar."
				);
			}

			layer(0).build(shape);
		
		} else {
			if (!layer(0).construida()) {
				throw new IllegalArgumentException(
					"\nÉ necessário que a primeira camada (" + layer(0).name() +
					") seja construída."
				);
			}
		}

		for (int i = 0; i < numLayers(); i++) {
			Layer camada = layer(i);
			
			if (i != 0) camada.build(layer(i-1).shapeOut());
			
			camada.initParams();
			camada.setId(i);
		}
		
		Dicionario dicio = new Dicionario();
		_loss = dicio.getLoss(loss);
		
		_optimizer = dicio.getOtimizador(otm);
		_optimizer.construir(params());
		
		_compiled = true;// modelo pode ser usado.
	}

	@Override
	public Layer layer(int id){
		if ((id < 0) || (id >= _camadas.length)) {
			throw new IllegalArgumentException(
				"O índice fornecido (" + id + 
				") é inválido ou fora de alcance."
			);
		}
	
		return _camadas[id];
	}

	@Override
	public Layer[] layers() {
		return _camadas;
	}

	@Override
	public Layer outputLayer() {
		if (_camadas.length < 1) {
			throw new UnsupportedOperationException(
				"\nO modelo não possui camadas adiciondas."
			);
		}

		return _camadas[_camadas.length-1];
	}

	@Override
	public float[] saidaParaArray() {
		validateCompilation();
		return outputLayer().output().array();
	}

	@Override
	public String name() {
		return name;
	}

	@Override
	public int numLayers() {
		return _camadas.length;
	}

	@Override
	protected String buildInfo() {
		final String pad = " ".repeat(4);
		StringBuilder sb = new StringBuilder();
		
		sb.append(name()).append(" [\n");

		int larguraId = String.valueOf(numLayers() - 1).length();

		sb.append(
			pad + String.format(
				"%-" + (larguraId + 3 + 16) + "s%-23s%-23s%-23s\n",
				"Camada", "Entrada", "Saída", "Parâmetros"
			)
		);

		for (Layer camada : this) {
			String idFormatado = String.format("%" + larguraId + "d", camada.id);
			String nomeCamada = idFormatado + " - " + camada.name();
			String formEntrada = JNNutils.arrayStr(camada.shapeIn());
			String formSaida   = JNNutils.arrayStr(camada.shapeOut());
			String parametros = String.format("%,d", camada.numParams());

			sb.append(
				pad + String.format(
					"%-" + (larguraId + 3 + 16) + "s%-23s%-23s%-23s\n",
					nomeCamada, formEntrada, formSaida, parametros
				)
			);
		}

		sb.append("\n");

		// Função de perda
		sb.append(pad).append("Perda: ").append(_loss.name());

		String params = String.format("%,d", numParams());
		sb.append("\n");
		sb.append(pad).append("Parâmetros: ").append(params).append("\n");
		
		long tamParams = 0;
		for (Layer camada : this) {
			if (camada.isTreinable()) {
				for (var param : camada.params()) {
					tamParams += param.weight.sizeBytes();
				}
			}
		}
		sb.append(pad).append("Tamanho Params: ").append(JNNutils.formatarTamBytes(tamParams)).append("\n");
		
		sb.append(pad).append("Tamanho Total: ").append(JNNutils.formatarTamBytes(sizeBytes())).append("\n");

		sb.append("]");

		return sb.toString();
	}

	@Override
	public void print() {
		validateCompilation();
		System.out.println(buildInfo());
	}

	@Override
	public Sequencial clone() {
		Sequencial clone = (Sequencial) super.clone();

		clone._evaluator = new Evaluator(clone);
		clone.trainHistory = this.trainHistory;
		clone.name = "Clone de " + name();
		
		Dicionario dicio = new Dicionario();
		clone._optimizer = dicio.getOtimizador(_optimizer.nome());
		clone._loss = dicio.getLoss(_loss.name());
		clone._trainer = new Treinador(clone);
		
		int nCamadas = numLayers();
		clone._camadas = new Layer[nCamadas];
		for (int i = 0; i < nCamadas; i++) {
			clone._camadas[i] = layer(i).clone();
		}
		clone._compiled = this._compiled;

		return clone;
	}

	@Override
	public Iterator<Layer> iterator() {
		return new ModelIterator(_camadas);
	}

	@Override
	public long sizeBytes() {
		long tam = 0;

		for (Layer c : this) {
			tam += c.sizeBytes();
		}

		return tam;
	}

}
