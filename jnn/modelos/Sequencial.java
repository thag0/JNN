package jnn.modelos;

import java.util.Iterator;

import jnn.camadas.Camada;
import jnn.camadas.Entrada;
import jnn.core.Dicionario;
import jnn.core.JNNutils;
import jnn.metrica.Avaliador;
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
public class Sequencial extends Modelo {

	/**
	 * Lista de camadas do modelo.
	 */
	private Camada[] _camadas;

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
		_camadas = new Camada[0];
		_compilado = false;
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
	public Sequencial(Camada... camadas) {
		this();// evitar repetição de código

		JNNutils.validarNaoNulo(camadas, "camadas == null.");

		if (camadas.length < 1) {
			throw new IllegalArgumentException(
				"\nConjunto de camadas vazio."
			);
		}
		
		add(camadas[0]);
		for (int i = 1; i < camadas.length; i++) {
			if (!(camadas[i] instanceof Entrada)) add(camadas[i]);
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
	public void add(Camada c) {
		_camadas = JNNutils.addEmArray(_camadas, c);
		_compilado = false;
	}

	/**
	 * Remove a última camada contida na lista de camadas do modelo.
	 * @return camada removida.
	 * @throws IllegalArgumentException caso o modelo já não possua nenhuma 
	 * camada disponível.
	 */
	public Camada rem() {
		if (_camadas.length < 1) {
			throw new IllegalArgumentException(
				"\nNão há camadas no modelo."
			);
		}

		Camada ultima = camadaSaida();

		Camada[] novas = _camadas;
		_camadas = new Camada[_camadas.length-1];
		System.arraycopy(novas, 0, _camadas, 0, _camadas.length);

		_compilado = false;

		return ultima;
	}

	@Override
	public void compilar(Object otm, Object loss) {
		if (camadas().length < 1) {
			throw new IllegalStateException(
				"\nNão há camadas no modelo."
			);
		}

		if (camada(0) instanceof Entrada) {
			int[] shape = camada(0).shapeOut();

			// remover camada de entrada do modelo
			Camada[] temp = camadas();
			_camadas = new Camada[numCamadas()-1];
			System.arraycopy(temp, 1, _camadas, 0, numCamadas());

			if (numCamadas() < 1) {
				throw new IllegalStateException(
					"\nO modelo não possui camadas para compilar."
				);
			}

			camada(0).construir(shape);
		
		} else {
			if (!camada(0)._construida) {
				throw new IllegalArgumentException(
					"\nÉ necessário que a primeira camada (" + camada(0).nome() +
					") seja construída."
				);
			}
		}

		for (int i = 0; i < numCamadas(); i++) {
			Camada camada = camada(i);
			
			if (i != 0) camada.construir(camada(i-1).shapeOut());
			
			camada.inicializar();
			camada.setId(i);
		}

		if (seedInicial != 0) _treinador.setSeed(seedInicial);
		
		Dicionario dicio = new Dicionario();
		_perda = dicio.getPerda(loss);
		
		_otimizador = dicio.getOtimizador(otm);
		_otimizador.construir(params(), grads());
		
		_compilado = true;// modelo pode ser usado.
	}

	@Override
	public Camada camada(int id){
		if ((id < 0) || (id >= _camadas.length)) {
			throw new IllegalArgumentException(
				"O índice fornecido (" + id + 
				") é inválido ou fora de alcance."
			);
		}
	
		return _camadas[id];
	}

	@Override
	public Camada[] camadas() {
		return _camadas;
	}

	@Override
	public Camada camadaSaida() {
		if (_camadas.length < 1) {
			throw new UnsupportedOperationException(
				"\nO modelo não possui camadas adiciondas."
			);
		}

		return _camadas[_camadas.length-1];
	}

	@Override
	public float[] saidaParaArray() {
		validarCompilacao();
		return camadaSaida().saida().array();
	}

	@Override
	public String nome() {
		return nome;
	}

	@Override
	public int numCamadas() {
		return _camadas.length;
	}

	@Override
	protected String construirInfo() {
		final String pad = " ".repeat(4);
		StringBuilder sb = new StringBuilder();
		
		sb.append(nome()).append(" [\n");

		int larguraId = String.valueOf(numCamadas() - 1).length();

		sb.append(
			pad + String.format(
				"%-" + (larguraId + 3 + 16) + "s%-23s%-23s%-23s\n",
				"Camada", "Entrada", "Saída", "Parâmetros"
			)
		);

		for (Camada camada : this) {
			String idFormatado = String.format("%" + larguraId + "d", camada.id);
			String nomeCamada = idFormatado + " - " + camada.nome();
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
		sb.append(pad).append("Perda: ").append(_perda.nome());

		String params = String.format("%,d", numParams());
		sb.append("\n");
		sb.append(pad).append("Parâmetros: ").append(params).append("\n");
		sb.append(pad).append("Tamanho: ").append(JNNutils.formatarTamBytes(tamBytes()));
		sb.append("\n").append("]");

		return sb.toString();
	}

	@Override
	public void print() {
		validarCompilacao();
		System.out.println(construirInfo());
	}

	@Override
	public Sequencial clone() {
		Sequencial clone = (Sequencial) super.clone();

		clone._avaliador = new Avaliador(clone);
		clone.calcularHistorico = this.calcularHistorico;
		clone.nome = "Clone de " + nome();
		
		Dicionario dicio = new Dicionario();
		clone._otimizador = dicio.getOtimizador(_otimizador.nome());
		clone._perda = dicio.getPerda(_perda.nome());
		clone.seedInicial = this.seedInicial;
		clone._treinador = new Treinador(clone);
		
		int nCamadas = numCamadas();
		clone._camadas = new Camada[nCamadas];
		for (int i = 0; i < nCamadas; i++) {
			clone._camadas[i] = camada(i).clone();
		}
		clone._compilado = this._compilado;

		return clone;
	}

	@Override
	public Iterator<Camada> iterator() {
		return new ModelIterator(_camadas);
	}

	@Override
	public long tamBytes() {
		long tam = 0;

		for (Camada c : this) {
			tam += c.tamBytes();
		}

		return tam;
	}

}
