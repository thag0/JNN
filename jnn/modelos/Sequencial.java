package jnn.modelos;

import jnn.avaliacao.Avaliador;
import jnn.avaliacao.perda.Perda;
import jnn.camadas.Camada;
import jnn.camadas.Entrada;
import jnn.core.Dicionario;
import jnn.core.tensor.Tensor;
import jnn.core.tensor.Variavel;
import jnn.otimizadores.Otimizador;
import jnn.treinamento.Treinador;

/**
 * <h1>
 *    Modelo sequencial de camadas
 * </h1>
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

		utils.validarNaoNulo(camadas, "Conjunto de camadas nulo.");

		if (camadas.length < 1) {
			throw new IllegalArgumentException(
				"\nConjunto de camadas vazio."
			);
		}

		for (int i = 0; i < camadas.length; i++) {
			utils.validarNaoNulo(camadas[i], ("Camada nula, id = " + i + "."));
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
	 * @param camada nova camada.
	 * @throws IllegalArgumentException se a camada fornecida for nula,
	 */
	public void add(Camada camada) {
		utils.validarNaoNulo(camada, "Camada nula.");
		_camadas = utils.addEmArray(_camadas, camada);
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
	public void compilar(Object otimizador, Object perda) {
		if (camadas().length < 1) {
			throw new IllegalStateException(
				"\nNão há camadas no modelo."
			);
		}

		int[] formato = {};

		if (_camadas[0] instanceof Entrada) {
			formato = _camadas[0].shapeSaida();

			//remover camada de entrada do modelo
			Camada[] temp = _camadas;
			_camadas = new Camada[_camadas.length-1];
			System.arraycopy(temp, 1, _camadas, 0, _camadas.length);

			if (_camadas.length == 0) {
				throw new IllegalStateException(
					"\nO modelo não possui camadas para compilar."
				);
			}

			_camadas[0].construir(formato);
		
		} else {
			if (!_camadas[0]._construida) {
				throw new IllegalArgumentException(
					"\nÉ necessário que a primeira camada (" + _camadas[0].nome() +
					") seja construída."
				);
			}
		}

		for (int i = 0; i < _camadas.length; i++) {
			_camadas[i].setId(i);

			if (i != 0) _camadas[i].construir(_camadas[i-1].shapeSaida());
			if (seedInicial != 0) _camadas[i].setSeed(seedInicial);

			_camadas[i].inicializar();
		}

		if (seedInicial != 0) _treinador.setSeed(seedInicial);
		
		Dicionario dicio = new Dicionario();
		_perda = dicio.getPerda(perda);
		
		_otimizador = dicio.getOtimizador(otimizador);
		_otimizador.construir(params(), grads());
		
		_compilado = true;// modelo pode ser usado.
	}

	@Override
	public Tensor forward(Object entrada) {
		validarCompilacao();

		utils.validarNaoNulo(entrada, "Dados de entrada não podem ser nulos.");

		Tensor prev = _camadas[0].forward(entrada);
		for (int i = 1; i < _camadas.length; i++) {
			prev = _camadas[i].forward(prev);
		}

		return prev.clone();// preservar a saída do modelo
	}
  
	@Override
	public void zerarGrad() {
		for (Camada camada : camadas()) {
			if (camada.treinavel()) camada.zerarGrad();
		}
	}

	@Override
	public void treino(boolean treinando) {
		for (Camada camada : camadas()) {
			camada.setTreino(treinando);
		}
	}

	@Override
	public Otimizador otimizador() {
		validarCompilacao();
		return _otimizador;
	}

	@Override
	public Perda perda() {
		validarCompilacao();
		return _perda;
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
	public Tensor[] params() {
		Tensor[] params = new Tensor[0];

		for (Camada camada : camadas()) {
			if (camada.treinavel()) {
				params = utils.addEmArray(params, camada.kernel());
				if (camada.temBias()) {
					params = utils.addEmArray(params, camada.bias());
				}
			}
		}

		return params;
	}

	@Override
	public Tensor[] grads() {
		Tensor[] grads = new Tensor[0];

		for (Camada camada : camadas()) {
			if (camada.treinavel()) {
				grads = utils.addEmArray(grads, camada.gradKernel());
				if (camada.temBias()) {
					grads = utils.addEmArray(grads, camada.gradBias());
				}
			}
		}

		return grads;
	}

	@Override
	public Variavel[] saidaParaArray() {
		validarCompilacao();
		return camadaSaida().saidaParaArray();
	}

	@Override
	public String nome() {
		return nome;
	}

	@Override
	public int numParametros() {
		int params = 0;

		for (Camada camada : camadas()) {
			params += camada.numParametros();
		}

		return params;
	}

	@Override
	public int numCamadas() {
		return _camadas.length;
	}

	@Override
	protected String construirInfo() {
		final String pad = " ".repeat(4);
		StringBuilder sb = new StringBuilder();
		
		sb.append(nome()).append(" = [\n");

		//otimizador
		sb.append(_otimizador.info()).append("\n");

		//função de perda
		sb.append(pad).append("Perda: ").append(_perda.nome());
		sb.append("\n").append("\n");

		//camadas
		sb.append(
			pad + String.format(
			"%-23s%-23s%-23s%-23s%-23s\n", "Camada", "Entrada", "Saída", "Ativação", "Parâmetros"
			)
		);

		for (Camada camada : camadas()) {
			
			//identificador da camada
			String nomeCamada = camada.id + " - " + camada.nome();
			
			//formato de entrada
			String formEntrada = utils.shapeStr(camada.shapeEntrada());
			
			//formato de saída
			String formSaida = utils.shapeStr(camada.shapeSaida());

			//função de ativação
			String ativacao;
			try {
				ativacao = camada.ativacao().nome();
			} catch (Exception e) {
				ativacao = "-";
			}

			String parametros = String.valueOf(camada.numParametros());

			sb.append(
				pad + String.format(
					"%-23s%-23s%-23s%-23s%-23s\n", nomeCamada, formEntrada, formSaida, ativacao, parametros
				)
			);
		}

		String params = String.format("%,d", numParametros());
		sb.append("\n");
		sb.append(pad).append("Parâmetros treináveis: " + params);
		sb.append("\n").append("]\n");

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
		clone._treinador = new Treinador();
		
		int nCamadas = numCamadas();
		clone._camadas = new Camada[nCamadas];
		for (int i = 0; i < nCamadas; i++) {
			clone._camadas[i] = camada(i).clone();
		}
		clone._compilado = this._compilado;

		return clone;
	}
}
