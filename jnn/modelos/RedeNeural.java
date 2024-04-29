package jnn.modelos;

import jnn.ativacoes.Ativacao;
import jnn.avaliacao.Avaliador;
import jnn.avaliacao.perda.MSE;
import jnn.avaliacao.perda.Perda;
import jnn.camadas.Camada;
import jnn.camadas.Densa;
import jnn.core.Dicionario;
import jnn.core.Tensor4D;
import jnn.otimizadores.Otimizador;
import jnn.otimizadores.SGD;
import jnn.treinamento.Treinador;

/**
 * <h3>
 *    Modelo de Rede Neural {@code Multilayer Perceptron} criado do zero
 * </h3>
 *  Possui um conjunto de camadas densas sequenciais que propagam os dados de entrada.
 * <p>
 *    O modelo pode ser usado tanto para problemas de {@code regressão e classificação}, contando com 
 *    algoritmos de treino e otimizadores variados para ajudar na convergência e desempenho da rede para 
 *    problemas diversos.
 * </p>
 * <p>
 *    Possui opções de configuração para funções de ativação de camadas individuais, valor de alcance 
 *    máximo e mínimo na aleatorização dos pesos iniciais, inicializadores de pesos e otimizadores que 
 *    serão usados durante o treino. 
 * </p>
 * <p>
 *    Após configurar as propriedades da rede, o modelo precisará ser {@code compilado} para efetivamente 
 *    poder ser utilizado.
 * </p>
 * <p>
 *    As predições do modelo são feitas usando o método de {@code calcularSaida()} onde é especificada
 *    uma única amostra de dados ou uma seqûencia de amostras onde é retornado o resultado de predição
 *    da rede.
 * </p>
 * <p>
 *    Opções de avaliação e desempenho do modelo podem ser acessadas através do {@code avaliador} da
 *    Rede Neural, que contém implementação de funções de perda e métricas para o modelo.
 * </p>
 * @author Thiago Barroso, acadêmico de Engenharia da Computação pela Universidade Federal do Pará, 
 * Campus Tucuruí. Maio/2023.
 */
public class RedeNeural extends Modelo {
	
	/**
	 * Região crítica
	 * <p>
	 *    Conjunto de camadas densas (ou fully connected) da Rede Neural.
	 * </p>
	 */
	private Densa[] _camadas;

	/**
	 * Array contendo a arquitetura de cada camada dentro da Rede Neural.
	 * <p>
	 *    Cada elemento da arquitetura representa a quantidade de neurônios 
	 *    presente na camada correspondente.
	 * </p>
	 * <p>
	 *    A "camada de entrada" não é considerada camada, pois não é alocada na
	 *    rede, ela serve apenas de parâmetro para o tamanho de entrada da primeira
	 *    camada densa da Rede Neural.
	 * </p>
	 */
	private int[] _arq;

	/**
	 * Constante auxiliar que ajuda no controle do bias dentro da rede.
	 */
	private boolean bias = true;

	/**
	 * <p>
	 *    Cria uma instância de rede neural artificial. A arquitetura da rede será baseada de acordo 
	 *    com cada posição do array, cada valor contido nele representará a quantidade de neurônios da 
	 *    camada correspondente.
	 * </p> 
	 * <p>
	 *   Nenhum dos elementos de arquitetura deve ser menor do que 1.
	 * </p>
	 * <p>
	 *    Exemplo de uso:
	 * </p>
	 * <pre>
	 * int[] arq = {
	 *    1, //tamanho de entrada da rede
	 *    2, //neurônios da primeira camada
	 *    3  //neurônios da segunda camada
	 * };
	 * </pre>
	 * <p>
	 *    É obrigatório que a arquitetura tenha no mínimo dois elementos, um para a entrada e outro
	 *    para a saída da Rede Neural.
	 * </p>
	 * <p>
	 *    Após instanciar o modelo, é necessário compilar por meio da função {@code compilar()};
	 * </p>
	 * <p>
	 *    Certifique-se de configurar as propriedades da rede por meio das funções de configuração fornecidas 
	 *    para obter os melhores resultados na aplicação específica. Caso não seja usada nenhuma das funções de 
	 *    configuração, a rede será compilada com os valores padrão.
	 * </p>
	 * @author Thiago Barroso, acadêmico de Engenharia da Computação pela Universidade Federal do Pará, 
	 * Campus Tucuruí. Maio/2023.
	 * @param arq modelo de arquitetura específico da rede.
	 * @throws IllegalArgumentException se o array de arquitetura for nulo.
	 * @throws IllegalArgumentException se o array de arquitetura não possuir, pelo menos, dois elementos.
	 * @throws IllegalArgumentException se os valores fornecidos forem menores que um.
	 */
	public RedeNeural(int... arq) {
		utils.validarNaoNulo(arq, "A arquitetura fornecida não deve ser nula.");

		if (arq.length < 2) {
			throw new IllegalArgumentException(
				"A arquitetura fornecida deve conter no mínimo dois elementos (entrada e saída), tamanho recebido = " + arq.length
			);
		}

		for (int i = 0; i < arq.length; i++) {
			if (arq[i] < 1) {
				throw new IllegalArgumentException(
					"Os valores de arquitetura fornecidos não devem ser maiores que zero."
				);
			}
		}

		_arq = arq;
		_compilado = false;
	}

	/**
	 * Define se a Rede Neural usará um viés para seus pesos.
	 * <p>
	 *    O viés é um atributo adicional para cada neurônio que sempre emite um valor de 
	 *    saída constante. A presença de viés permite que a rede neural aprenda relações 
	 *    mais complexas, melhorando a capacidade de modelagem.
	 * </p>
	 * <p>
	 *    O bias deve ser configurado antes da compilação para ser aplicado.
	 * </p>
	 * <p>
	 *    {@code O valor padrão para uso do bias é true}
	 * </p>
	 * @param usarBias novo valor para o uso do bias.
	 */
	public void configurarBias(boolean usarBias) {
		this.bias = usarBias;
	}

	/**
	 * Configura a função de ativação de todas as camadas da rede. É preciso
	 * compilar o modelo previamente para poder configurar suas funções de ativação.
	 * <p>
	 *    Letras maiúsculas e minúsculas não serão diferenciadas.
	 * </p>
	 * <p>
	 *    Segue a lista das funções de ativação disponíveis:
	 * </p>
	 * <ul>
	 *    <li> ReLU. </li>
	 *    <li> Sigmoid. </li>
	 *    <li> TanH. </li>
	 *    <li> Leaky ReLU. </li>
	 *    <li> ELU .</li>
	 *    <li> Swish. </li>
	 *    <li> GELU. </li>
	 *    <li> Linear. </li>
	 *    <li> Seno. </li>
	 *    <li> Argmax. </li>
	 *    <li> Softmax. </li>
	 *    <li> Softplus. </li>
	 *    <li> ArcTan. </li>
	 * </ul>
	 * <p>
	 *    {@code A função de ativação padrão é a Linear para todas as camadas}
	 * </p>
	 * @param ativacao instância da função de ativação.
	 * @throws IllegalArgumentException se o modelo não foi compilado previamente.
	 */
	public void configurarAtivacao(Ativacao ativacao) {
		verificarCompilacao();
		
		for (Camada camada : _camadas) {
			camada.setAtivacao(ativacao);
		}
	}

	/**
	 * Configura a função de ativação de todas as camadas da rede. É preciso
	 * compilar o modelo previamente para poder configurar suas funções de ativação.
	 * <p>
	 *    Letras maiúsculas e minúsculas não serão diferenciadas.
	 * </p>
	 * <p>
	 *    Segue a lista das funções de ativação disponíveis:
	 * </p>
	 * <ul>
	 *    <li> ReLU. </li>
	 *    <li> Sigmoid. </li>
	 *    <li> TanH. </li>
	 *    <li> Leaky ReLU. </li>
	 *    <li> ELU .</li>
	 *    <li> Swish. </li>
	 *    <li> GELU. </li>
	 *    <li> Linear. </li>
	 *    <li> Seno. </li>
	 *    <li> Argmax. </li>
	 *    <li> Softmax. </li>
	 *    <li> Softplus. </li>
	 *    <li> ArcTan. </li>
	 * </ul>
	 * <p>
	 *    {@code A função de ativação padrão é a Linear para todas as camadas}
	 * </p>
	 * @param ativacao nome da função de ativação.
	 * @throws IllegalArgumentException se o modelo não foi compilado previamente.
	 */
	public void configurarAtivacao(String ativacao) {
		verificarCompilacao();
		
		for (Camada camada : _camadas) {
			camada.setAtivacao(ativacao);
		}
	}

	/**
	 * Configura a função de ativação da camada correspondente. É preciso
	 * compilar o modelo previamente para poder configurar suas funções de ativação.
	 * <p>
	 *    Letras maiúsculas e minúsculas não serão diferenciadas.
	 * </p>
	 * <p>
	 *    Segue a lista das funções de ativação disponíveis:
	 * </p>
	 * <ul>
	 *    <li> ReLU. </li>
	 *    <li> Sigmoid. </li>
	 *    <li> TanH. </li>
	 *    <li> Leaky ReLU. </li>
	 *    <li> ELU .</li>
	 *    <li> Swish. </li>
	 *    <li> GELU. </li>
	 *    <li> Linear. </li>
	 *    <li> Seno. </li>
	 *    <li> Argmax. </li>
	 *    <li> Softmax. </li>
	 *    <li> Softplus. </li>
	 *    <li> ArcTan. </li>
	 * </ul>
	 * <p>
	 *    {@code A função de ativação padrão é a Linear para todas as camadas}
	 * </p>
	 * @param camada camada que será configurada.
	 * @param ativacao instância da função de ativação.
	 * @throws IllegalArgumentException se o modelo não foi compilado previamente.
	 */
	public void configurarAtivacao(Densa camada, Ativacao ativacao) {
		utils.validarNaoNulo(camada, "A camada não pode ser nula.");
		camada.setAtivacao(ativacao);
	}

	/**
	 * Configura a função de ativação da camada correspondente. É preciso
	 * compilar o modelo previamente para poder configurar suas funções de ativação.
	 * <p>
	 *    Letras maiúsculas e minúsculas não serão diferenciadas.
	 * </p>
	 * <p>
	 *    Segue a lista das funções de ativação disponíveis:
	 * </p>
	 * <ul>
	 *    <li> ReLU. </li>
	 *    <li> Sigmoid. </li>
	 *    <li> TanH. </li>
	 *    <li> Leaky ReLU. </li>
	 *    <li> ELU .</li>
	 *    <li> Swish. </li>
	 *    <li> GELU. </li>
	 *    <li> Linear. </li>
	 *    <li> Seno. </li>
	 *    <li> Argmax. </li>
	 *    <li> Softmax. </li>
	 *    <li> Softplus. </li>
	 *    <li> ArcTan. </li>
	 * </ul>
	 * <p>
	 *    {@code A função de ativação padrão é a Linear para todas as camadas}
	 * </p>
	 * @param camada camada que será configurada.
	 * @param ativacao nome da função de ativação.
	 * @throws IllegalArgumentException se o modelo não foi compilado previamente.
	 */
	public void configurarAtivacao(Densa camada, String ativacao) {
		utils.validarNaoNulo(camada, "A camada não pode ser nula.");
		camada.setAtivacao(ativacao);
	}

	/**
	 * Compila o modelo de Rede Neural inicializando as camadas, neurônios e pesos respectivos, 
	 * baseado nos valores fornecidos.
	 * <p>
	 *    Caso nenhuma configuração inicial seja feita, a rede será inicializada com os argumentos padrão. 
	 * </p>
	 * Após a compilação o modelo está pronto para ser usado, mas deverá ser treinado.
	 * <p>
	 *    Para treinar o modelo deve-se fazer uso da função função {@code treinar()} informando os 
	 *    dados necessários para a rede.
	 * </p>
	 * <p>
	 *    Para usar as predições da rede basta usar a função {@code calcularSaida()} informando os
	 *    dados necessários. Após a predição pode-se obter o resultado da rede por meio da função 
	 *    {@code obterSaidas()};
	 * </p>
	 * Os valores de função de perda, otimizador serão definidos como os padrões 
	 * {@code ErroMedioQuadrado (MSE)} e {@code SGD}. 
	 * <p>
	 *    Valores de perda e otimizador configurados previamente são mantidos.
	 * </p>
	 */
	public void compilar() {
		//usando valores de configuração prévia, se forem criados.
		Otimizador o = (_otimizador == null) ? new SGD() : _otimizador;
		Perda p = (_perda == null) ? new MSE() : _perda;

		compilar(o, p);
	}

	/**
	 * Compila o modelo de Rede Neural inicializando as camadas, neurônios e pesos respectivos, 
	 * baseado nos valores fornecidos.
	 * <p>
	 *    Caso nenhuma configuração inicial seja feita, a rede será inicializada com os argumentos padrão. 
	 * </p>
	 * Após a compilação o modelo está pronto para ser usado, mas deverá ser treinado.
	 * <p>
	 *    Para treinar o modelo deve-se fazer uso da função função {@code treinar()} informando os 
	 *    dados necessários para a rede.
	 * </p>
	 * <p>
	 *    Para usar as predições da rede basta usar a função {@code calcularSaida()} informando os
	 *    dados necessários. Após a predição pode-se obter o resultado da rede por meio da função 
	 *    {@code obterSaidas()};
	 * </p>
	 * O valor do otimizador será definido como {@code SGD}.
	 * <p>
	 *    Valor de otimizador configurado previamente é mantido.
	 * </p>
	 * @param perda função de perda da Rede Neural usada durante o treinamento.
	 * @throws IllegalArgumentException se a função de perda for nula.
	 */
	public void compilar(Perda perda) {
		utils.validarNaoNulo(perda, "A função de perda não pode ser nula.");

		//usando valores de configuração prévia, se forem criados
		Otimizador o = (_otimizador == null) ? new SGD() : _otimizador;
		compilar(o, perda);
	}

	/**
	 * Compila o modelo de Rede Neural inicializando as camadas, neurônios e pesos respectivos, 
	 * baseado nos valores fornecidos.
	 * <p>
	 *    Caso nenhuma configuração inicial seja feita, a rede será inicializada com os argumentos padrão. 
	 * </p>
	 * Após a compilação o modelo está pronto para ser usado, mas deverá ser treinado.
	 * <p>
	 *    Para treinar o modelo deve-se fazer uso da função função {@code treinar()} informando os 
	 *    dados necessários para a rede.
	 * </p>
	 * <p>
	 *    Para usar as predições da rede basta usar a função {@code calcularSaida()} informando os
	 *    dados necessários. Após a predição pode-se obter o resultado da rede por meio da função 
	 *    {@code obterSaidas()};
	 * </p>
	 * A função de perda usada será a {@code ErroMedioQuadrado (MSE)}.
	 * <p>
	 *    Valor de função de perda configurada previamente é mantido.
	 * </p>
	 * @param otimizador otimizador que será usando para o treino da Rede Neural.
	 * @throws IllegalArgumentException se o otimizador ou inicializador forem nulos.
	 */
	public void compilar(Otimizador otimizador) {
		utils.validarNaoNulo(otimizador, "O otimizador fornecido não pode ser nulo.");

		//usando valores de configuração prévia, se forem criados.
		Perda p = (_perda == null) ? new MSE() : _perda;
		compilar(otimizador, p);
	}

	@Override
	public void compilar(Object otimizador, Object perda) {
		_camadas = new Densa[_arq.length-1];
		_camadas[0] = new Densa(_arq[1]);
		_camadas[0].setBias(bias);
		_camadas[0].construir(new int[]{_arq[0]});

		Dicionario dic = new Dicionario();
		for (int i = 1; i < _camadas.length; i++) {
			_camadas[i] = new Densa(_arq[i+1]);
			_camadas[i].setBias(bias);
			_camadas[i].construir(_camadas[i-1].formatoSaida());
		}

		for (int i = 0; i < _camadas.length; i++) {
			_camadas[i].setId(i);
			if (seedInicial != 0) _camadas[i].setSeed(seedInicial);
			_camadas[i].inicializar();
		}

		if (seedInicial != 0) _treinador.setSeed(seedInicial);

		_perda = dic.getPerda(perda);
		_otimizador = dic.getOtimizador(otimizador);

		_otimizador.construir(_camadas);

		_compilado = true;
	}

	/**
	 * Alimenta os dados pela rede neural usando o método de feedforward através do conjunto
	 * de dados fornecido. 
	 * <p>
	 *    Os dados são alimentados para as entradas dos neurônios e é calculado o produto junto 
	 *    com os pesos. No final é aplicado a função de ativação da camada no neurônio e o resultado 
	 *    fica armazenado na saída dele.
	 * </p>
	 * @param entrada dados usados para alimentar a camada de entrada.
	 * @throws IllegalArgumentException se o modelo não foi compilado previamente.
	 * @throws IllegalArgumentException se o tamanho dos dados de entrada for diferente da capacidade
	 * de entrada da rede.
	 */
	@Override
	public Tensor4D forward(Object entrada) {
		verificarCompilacao();

		utils.validarNaoNulo(entrada, "Dados de entrada não pode ser nulo.");

		Tensor4D prev = _camadas[0].forward(entrada);
		for (int i = 1; i < _camadas.length; i++) {
			prev = _camadas[i].forward(prev);
		}

		return prev.clone();//preservar a saída do modelo
	}

	/**
	 * Alimenta os dados pela rede neural usando o método de feedforward através do conjunto
	 * de dados fornecido. 
	 * <p>
	 *    Os dados são alimentados para as entradas dos neurônios e é calculado o produto junto 
	 *    com os pesos. No final é aplicado a função de ativação da camada no neurônio e o resultado 
	 *    fica armazenado na saída dele.
	 * </p>
	 */
	@Override
	public Tensor4D[] forwards(Object[] entradas) {
		verificarCompilacao();

		utils.validarNaoNulo(entradas, "Dados de entrada não podem ser nulos.");
		
		final int tam = entradas.length;
		int numThreads = Runtime.getRuntime().availableProcessors();
		if (numThreads > tam) numThreads = tam;

		Tensor4D[] prevs = new Tensor4D[tam];
		RedeNeural[] clones = new RedeNeural[numThreads];
		Thread[] threads = new Thread[numThreads];

		for (int i = 0; i < numThreads; i++) {
			clones[i] = clone();
		}

		int lote = tam / numThreads;
		for (int i = 0; i < numThreads; i++) {
			final int id = i;
			final int inicio = i * lote;
			final int fim = (i == numThreads - 1) ? tam : (i + 1) * lote;
	
			threads[id] = new Thread(() -> {
				for (int j = inicio; j < fim; j++) {
					prevs[j] = clones[id].forward(entradas[j]);
				}
			});
			threads[i].start();
		}

		try {
			for (Thread t : threads) {
				t.join();
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
			System.exit(1);
		}

		return prevs;
	}

	@Override
	public void zerarGradientes() {
		for (int i = 0; i < _camadas.length; i++) {
			_camadas[i].zerarGradientes();
		}
	}

	@Override
	public void treino(boolean treinando) {
		for (Camada camada : _camadas) {
			camada.setTreino(treinando);
		}
	}

	@Override
	public Otimizador otimizador() {
		return _otimizador;
	}

	@Override
	public Perda perda() {
		return _perda;
	}

	/**
	 * Retorna a {@code camada} da Rede Neural correspondente ao índice fornecido.
	 * @param id índice da busca.
	 * @return camada baseada na busca.
	 * @throws IllegalArgumentException se o modelo não foi compilado previamente.
	 * @throws IllegalArgumentException se o índice estiver fora do alcance do tamanho 
	 * das camadas ocultas.
	 */
	@Override
	public Densa camada(int id) {
		verificarCompilacao();

		if ((id < 0) || (id >= _camadas.length)) {
			throw new IllegalArgumentException(
				"O índice fornecido (" + id + 
				") é inválido ou fora de alcance."
			);
		}
	
		return _camadas[id];
	}

	/**
	 * Retorna todo o conjunto de camadas densas presente na Rede Neural.
	 * @throws IllegalArgumentException se o modelo não foi compilado previamente.
	 * @return conjunto de camadas da rede.
	 */
	@Override
	public Densa[] camadas() {
		verificarCompilacao();
		return _camadas;
	}

	/**
	 * Retorna a {@code camada de saída} da Rede Neural.
	 * @return camada de saída, ou ultima camada densa.
	 * @throws IllegalArgumentException se o modelo não foi compilado previamente.
	 */
	@Override
	public Densa camadaSaida() {
		verificarCompilacao();
		return _camadas[_camadas.length-1];
	}

	/**
	 * Retorna os dados de saída da última camada da Rede Neural. 
	 * <p>
	 *    A ordem de cópia é crescente, do primeiro neurônio da saída ao último.
	 * </p>
	 * @return array com os dados das saídas da rede.
	 * @throws IllegalArgumentException se o modelo não foi compilado previamente.
	 */
	@Override
	public double[] saidaParaArray() {
		verificarCompilacao();
		return camadaSaida().saidaParaArray();
	}

	/**
	 * Retorna o array que representa a estrutura da Rede Neural. Nele cada elemento 
	 * indica uma camada da rede e cada valor contido nesse elemento indica a 
	 * quantidade de neurônios daquela camada correspondente.
	 * <p>
	 *    Nessa estrutura de rede, a camada de entrada não é considerada uma camada,
	 *    o que significa dizer também que ela não é uma instância de camada dentro
	 *    da Rede Neural.
	 * </p>
	 * <p>
	 *    A "camada de entrada" representa o tamanho de entrada da primeira camada densa
	 *    da rede, ou seja, ela é apenas um parâmetro pro tamanho de entrada da primeira
	 *    camada oculta. 
	 * </p>
	 * @return array com a arquitetura da rede.
	 * @throws IllegalArgumentException se o modelo não foi compilado previamente.
	 */
	public int[] obterArquitetura() {
		verificarCompilacao();
		return _arq;
	}

	/**
	 * Informa o nome configurado da Rede Neural.
	 * @return nome específico da rede.
	 */
	@Override
	public String nome() {
		return this.nome;
	}

	/**
	 * Retorna a quantidade total de parâmetros da rede.
	 * <p>
	 *    isso inclui todos os pesos de todos os neurônios presentes 
	 *    (incluindo o peso adicional do bias).
	 * </p>
	 * @return quantiade de parâmetros total da rede.
	 */
	@Override
	public int numParametros() {
		int params = 0;
		for (Camada camada : _camadas) {
			params += camada.numParametros();
		}

		return params;
	}

	/**
	 * Retorna a quantidade de camadas densas presente na Rede Neural.
	 * <p>
	 *    A {@code camada de entrada} não é considerada uma camada densa e é usada
	 *    apenas para especificar o tamanho de entrada suportado pela rede.
	 * </p>
	 * @return quantidade de camadas da Rede Neural.
	 * @throws IllegalArgumentException se o modelo não foi compilado previamente.
	 */
	@Override
	public int numCamadas() {
		verificarCompilacao();
		return _camadas.length;
	}

	/**
	 * Retorna a capacidade da camada de entrada da Rede Neural. Em outras palavras
	 * diz quantos dados de entrada a rede suporta.
	 * @return tamanho de entrada da Rede Neural.
	 * @throws IllegalArgumentException se o modelo não foi compilado previamente.
	 */
	public int obterTamanhoEntrada() {
		verificarCompilacao();
		return _arq[0];
	}

	/**
	 * Retorna a capacidade de saída da Rede Neural. Em outras palavras
	 * diz quantos dados de saída a rede produz.
	 * @return tamanho de saída da Rede Neural.
	 * @throws IllegalArgumentException se o modelo não foi compilado previamente.
	 */
	public int obterTamanhoSaida() {
		verificarCompilacao();
		return _arq[_arq.length-1];
	}

	/**
	 * Retorna o valor de uso do bias da Rede Neural.
	 * @return valor de uso do bias da Rede Neural.
	 */
	public boolean temBias() {
		return bias;
	}

	/**
	 * Disponibiliza o histórico da função de perda da Rede Neural durante cada época
	 * de treinamento.
	 * <p>
	 *    O histórico será o do ultimo processo de treinamento usado, seja ele sequencial ou em
	 *    lotes. Sendo assim, por exemplo, caso o treino seja em sua maioria feito pelo modo sequencial
	 *    mas logo depois é usado o treino em lotes, o histórico retornado será o do treinamento em lote.
	 * </p>
	 * @return lista contendo o histórico de perdas durante o treinamento da rede.
	 * @throws IllegalArgumentException se não foi habilitado previamente o cálculo do 
	 * histórico de custos.
	 */
	@Override
	public double[] historico() {
		if (_treinador.calcularHistorico) {
			return _treinador.historico();   
		
		} else {
			throw new UnsupportedOperationException(
				"O histórico de treino da rede deve ser configurado previamente."
			);
		}
	}

	@Override
	protected String construirInfo() {
		StringBuilder sb = new StringBuilder();
		String pad = "    ";
		System.out.println(nome() + " = [");

		//otimizador
		sb.append(_otimizador.info()).append("\n");

		//perda
		sb.append(pad + "Perda: " + _perda.nome() + "\n\n");

		//bias
		sb.append(pad + "Bias = " + bias);
		sb.append("\n\n");

		//ativações
		for (int i = 0; i < _camadas.length; i++) {
			sb.append(
				pad + "Ativação camada " + i + ": " + 
				_camadas[i].ativacao().nome() + "\n"
			);
		}

		//arquitetura
		sb.append("\n" + pad + "arquitetura = (" + _arq[0]);
		for (int i = 1; i < _arq.length; i++) {
			sb.append(", " + _arq[i]);
		}
		sb.append(")\n");

		sb.append(pad).append("Parâmetros: ").append(numParametros());

		sb.append("\n]\n");
		
		return sb.toString();
	}

	/**
	 * Exibe algumas informações importantes sobre a Rede Neural, como:
	 * <ul>
	 *    <li>
	 *       Otimizador atual e suas informações específicas.
	 *    </li>
	 *    <li>
	 *       Contém bias adicionado nas camadas.
	 *    </li>
	 *    <li>
	 *       Função de ativação de todas as camadas.
	 *    </li>
	 *    <li>
	 *       Arquitetura da rede.
	 *    </li>
	 * </ul>
	 * @return buffer formatado contendo as informações.
	 * @throws IllegalArgumentException se o modelo não foi compilado previamente.
	 */
	@Override
	public void info() {
		verificarCompilacao();
		System.out.println(construirInfo());
	}

	@Override
	public RedeNeural clone() {
		RedeNeural clone = (RedeNeural) super.clone();

		clone._avaliador = new Avaliador(clone);
		clone.calcularHistorico = this.calcularHistorico;
		clone.nome = "Clone de " + nome();

		clone._arq = this._arq.clone();
		clone.bias = this.bias;

		Dicionario dicio = new Dicionario();
		clone._otimizador = dicio.getOtimizador(_otimizador.nome());
		clone._perda = dicio.getPerda(_perda.nome());
		clone.seedInicial = this.seedInicial;
		clone._treinador = new Treinador();

		clone._camadas = new Densa[_camadas.length];
		for (int i = 0; i < _camadas.length; i++) {
			clone._camadas[i] = _camadas[i].clone();
		}
		clone._compilado = this._compilado;

		return clone;
	}
}
