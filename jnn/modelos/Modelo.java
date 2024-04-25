package jnn.modelos;

import jnn.avaliacao.Avaliador;
import jnn.avaliacao.perda.Perda;
import jnn.camadas.Camada;
import jnn.core.Tensor4D;
import jnn.core.Utils;
import jnn.otimizadores.Otimizador;
import jnn.treinamento.Treinador;

/**
 * <h3>
 *    Basa para crianção de modelos dentro da biblioteca.
 * </h3>
 * Contém a inteface para os métodos necessários que são usados
 * para implementação de modelos.
 */
public abstract class Modelo implements Cloneable {

	/**
	 * Nome da instância do modelo.
	 */
	protected String nome = getClass().getSimpleName();

	/**
	 * <h3> Não alterar </h3>
	 * Auxiliar no controle da compilação do modelo, ajuda a evitar uso 
	 * indevido caso ainda não tenha suas variáveis e dependências inicializadas 
	 * previamente.
	 */
	public boolean _compilado;

	/**
	 * Função de perda para avaliar o erro durante o treino.
	 */
	protected Perda _perda;

	/**
	 * Otimizador usado para ajuste de parâmetros treináveis.
	 */
	protected Otimizador _otimizador;

	/**
	 * Ponto inicial para os geradores aleatórios.
	 * <p>
	 *    0 não é considerado uma seed e ela só é configurada quando esse valor é
	 *    alterado.
	 * </p>
	 */
	protected long seedInicial = 0;

	/**
	 * Gerenciador de treino do modelo. contém implementações dos 
	 * algoritmos de treino para o ajuste de parâmetros treináveis.
	 */
	protected Treinador _treinador;

	/**
	 * Auxiliar na verificação para o salvamento do histórico
	 * de perda do modelo durante o treinamento.
	 */
	protected boolean calcularHistorico = false;

	/**
	 * Responsável pelo retorno de desempenho do modelo.
	 * Contém implementações de métodos tanto para cálculo de perdas
	 * quanto de métricas.
	 * <p>
	 *    Cada modelo possui seu próprio avaliador.
	 * </p>
	 */
	protected Avaliador _avaliador;

	/**
	 * Utilitário.
	 */
	Utils utils;
	
	/**
	 * Inicialização implicita de um modelo.
	 */
	protected Modelo() {
		_treinador = new Treinador();
		_avaliador = new Avaliador(this);
		utils = new Utils();
	}

	/**
	 * <p>
	 *    Altera o nome do modelo.
	 * </p>
	 * O nome é apenas estético e não influencia na performance ou na 
	 * usabilidade do modelo.
	 * @param nome novo nome da rede.
	 */
	public void setNome(String nome) {
		if (nome != null) {
			String s = nome.trim();
			if (!s.isEmpty()) this.nome = s;
		}
	}

	/**
	 * Configura a nova seed inicial para os geradores de números aleatórios utilizados 
	 * durante o processo de inicialização de parâmetros treináveis do modelo.
	 * <p>
	 *    Configurações personalizadas de seed permitem fazer testes com diferentes
	 *    parâmetros, buscando encontrar um melhor ajuste para o modelo.
	 * </p>
	 * <p>
	 *    A configuração de seed deve ser feita antes da compilação do modelo para
	 *    surtir efeito.
	 * </p>
	 * @param seed nova seed.
	 */
	public void setSeed(long seed) {
		seedInicial = seed;
	}

	/**
	 * Define se, durante o processo de treinamento, o modelo irá salvar os dados 
	 * relacionados a função de perda de cada época.
	 * <p>
	 *    Calcular a perda é uma operação que pode ser computacionalmente cara 
	 *    dependendo do tamanho do modelo e do conjunto de dados, então deve ser 
	 *    bem avaliado querer habilitar ou não esse recurso.
	 * </p>
	 * <p>
	 *    {@code O valor padrão é false}
	 * </p>
	 * @param calc se verdadeiro, o modelo armazenará o histórico de perda 
	 * durante cada época de treinamento.
	 */
	public void setHistorico(boolean calc) {
		calcularHistorico = calc;
		_treinador.setHistorico(calc);
	}

	/**
	 * Configura a função de perda que será utilizada durante o processo
	 * de treinamento do modelo.
	 * @param perda nova função de perda.
	 */
	public void set_perda(Perda perda) {
		utils.validarNaoNulo(perda, "A função de perda não pode ser nula.");

		this._perda = perda;
	}

	/**
	 * Configura o novo otimizador do modelo com base numa nova instância 
	 * de otimizador.
	 * <p>
	 *    Configurando o otimizador informando diretamente uma nova instância 
	 *    permite configurar os hiperparâmetros do otimizador fora dos valores 
	 *    padrão, o que pode ajudar a melhorar o desempenho de aprendizado do 
	 *    modelo em cenários específicos.
	 * </p>
	 * Otimizadores disponíveis.
	 * <ol>
	 *    <li> GD (Gradient Descent) </li>
	 *    <li> SGD (Stochastic Gradient Descent) </li>
	 *    <li> AdaGrad </li>
	 *    <li> RMSProp </li>
	 *    <li> Adam  </li>
	 *    <li> Nadam </li>
	 *    <li> AMSGrad </li>
	 *    <li> Adadelta </li>
	 * </ol>
	 * @param otm novo otimizador.
	 */
	public void set_otimizador(Otimizador otm) {
		utils.validarNaoNulo(otm, "O novo otimizador não pode ser nulo.");

		_otimizador = otm;
	}

	/**
	 * Inicializa os parâmetros necessários para cada camada do modelo,
	 * além de gerar os valores iniciais para os kernels e bias.
	 * <p>
	 *    Caso nenhuma configuração inicial seja feita ou sejam fornecidos 
	 *    apenas nomes referenciando os objetos desejados, o modelo será 
	 *    compilado com os valores padrões. 
	 * </p>
	 * <p>
	 *    Otimizadores podem ser recebidos usando instâncias pré configuradas, 
	 *    essas intâncias dão a liberdade de inicializar o otimizador com valores
	 *    personalizáveis para seus parâmetros (como taxa de aprendizagem, por exemplo).
	 * </p>
	 * <p>
	 *    Para treinar o modelo deve-se fazer uso da função função {@code treinar()} 
	 *    informando os dados necessários para treino.
	 * </p>
	 * @param otimizador otimizador usando para ajustar os parâmetros treinavéis do 
	 * modelo, pode ser uma {@code String} referente ao nome ou uma {@code instância} 
	 * já inicializada.
	 * @param perda função de perda usada para avaliar o erro do modelo durante o 
	 * treino, pode ser uma {@code String} referente ao nome ou uma {@code instância} 
	 * já inicializada.
	 */
	public abstract void compilar(Object otimizador, Object perda);

	/**
	 * Auxiliar na verificação da compilação do modelo.
	 */
	protected void verificarCompilacao() {
		if (!_compilado) {
			throw new IllegalStateException(
				"\nO modelo ainda não foi compilado."
			);
		}
	}

	/**
	 * Alimenta o modelo com os dados de entrada.
	 * @param entrada dados de entrada que serão propagados através do modelo.
	 * @return {@code Tensor} contendo a saída prevista pelo modelo.
	 */
	public abstract Tensor4D forward(Object entrada);

	/**
	 * Alimenta o modelo com vários dados de entrada.
	 * @param entradas array contendo multiplas entradas para testar o modelo.
	 * @return array de {@code Tensor} contendo as previsões correspondentes.
	 */
	public abstract Tensor4D[] forwards(Object[] entradas);

	/**
	 * Zera os gradientes acumulados do modelo.
	 * <p>
	 *    Apenas gradientes de camadas treináveis serão zerados.
	 * </p>
	 */
	public abstract void zerarGradientes();

	/**
	 * Treina o modelo de acordo com as configurações predefinidas.
	 * @param entradas dados de entrada do treino (features).
	 * @param saidas dados de saída correspondente a entrada (classes).
	 * @param epochs quantidade de épocas de treinamento.
	 * @param logs logs para perda durante as épocas de treinamento.
	 */
	public void treinar(Object entradas, Object[] saidas, int epochs, boolean logs) {
		verificarCompilacao();

		utils.validarNaoNulo(entradas, "Dados de entrada não podem ser nulos.");
		utils.validarNaoNulo(saidas, "Dados de saida não podem ser nulos.");

		if (epochs < 1) {
			throw new IllegalArgumentException(
				"\nO valor de épocas deve ser maior que zero, recebido = " + epochs
			);
		}

		_treinador.treino(this, entradas, saidas, epochs, logs);
	}
	
	/**
	 * Treina o modelo de acordo com as configurações predefinidas utilizando o
	 * treinamento em lotes.
	 * @param entradas dados de entrada do treino (features).
	 * @param saidas dados de saída correspondente a entrada (class).
	 * @param epochs quantidade de épocas de treinamento.
	 * @param tamLote tamanho do lote de treinamento.
	 * @param logs logs para perda durante as épocas de treinamento.
	 */
	public void treinar(Object entradas, Object[] saidas, int epochs, int tamLote, boolean logs) {
		verificarCompilacao();

		utils.validarNaoNulo(entradas, "Dados de entrada não podem ser nulos.");
		utils.validarNaoNulo(saidas, "Dados de saida não podem ser nulos.");

		if (epochs < 1) {
			throw new IllegalArgumentException(
				"\nO valor de epochs (" + epochs + ") não pode ser menor que um"
			);
		}

		_treinador.treino(this, entradas, saidas, epochs, tamLote, logs);
	}

	/**
	 * Configura o modelo para o modo de treino
	 * @param treinando {@code true} caso o modelo deva estar no modo treino,
	 * {@code false} caso contrário.
	 */
	public abstract void treino(boolean treinando);

	/**
	 * Avalia o modelo, calculando o seu valor de perda fazendo uso da função 
	 * de perda que foi configurada.
	 * <p>
	 *    É possível utilizar outras funções de perda mesmo que sejam diferentes
	 *    da que o modelo usa, através de:
	 * </p>
	 * <pre>
	 * modelo.avaliador()
	 * </pre>
	 * @param entrada dados de entrada para avaliação.
	 * @param saida dados de saída correspondente as entradas fornecidas.
	 * @return valor de perda do modelo.
	 */
	public double avaliar(Object entrada, Object[] saida) {
		verificarCompilacao();

		utils.validarNaoNulo(entrada, "Dados de entrada não podem ser nulos.");
		utils.validarNaoNulo(saida, "Dados de saida não podem ser nulos.");

		//por enquanto uma instância local
		Object[] amostras = utils.transformarParaArray(entrada);
 
		if (amostras.length != saida.length) {
			throw new IllegalArgumentException(
				"\nA quantidade de dados de entrada (" + amostras.length + ") " +
				"e saída (" + saida.length + ") " + "devem ser iguais."
			);
		}
 
		if (saida instanceof double[][] == false) {
			throw new IllegalArgumentException(
				"\nA saída deve ser do tipo double[][]" + 
				" recebido " + saida.getClass().getTypeName()
			);
		}

		double[][] s = (double[][]) saida;
		int n = amostras.length;
		double soma = 0;
		for (int i = 0; i < n; i++) {
			forward(amostras[i]);
			soma += _perda.calcular(saidaParaArray(), s[i]);
		}

		return soma/n;
	}

	/**
	 * Retorna o avaliador do modelo, 
	 * <p>
	 *    O avaliador contém diferentes métodos de métricas úteis
	 *    para medir seu desempenho.
	 * </p>
	 * @return avaliador do modelo.
	 */
	public Avaliador avaliador() {
		return this._avaliador;
	}

	/**
	 * Retorna o otimizador configurado para o treino do modelo modelo.
	 * @return otimizador atual do modelo.
	 */
	public abstract Otimizador otimizador();

	/**
	 * Retorna a função de perda configurada do modelo.
	 * @return função de perda atual do modelo.
	 */
	public abstract Perda perda();

	/**
	 * Retorna a {@code camada} do Modelo correspondente ao índice fornecido.
	 * @param id índice da busca.
	 * @return camada baseada na busca.
	 */
	public abstract Camada camada(int id);

	/**
	 * Retorna todo o conjunto de camadas presente no modelo.
	 * @return conjunto de camadas do modelo.
	 */
	public abstract Camada[] camadas();

	/**
	 * Retorna a {@code camada de saída}, ou última camada, do modelo.
	 * @return camada de saída.
	 */
	public abstract Camada camadaSaida();
	
	/**
	 * Retorna um array contendo a saída serializada do modelo.
	 * @return saída do modelo.
	 */
	public abstract double[] saidaParaArray();

	/**
	 * Copia os dados de saída da última camada do modelo para o array.
	 * @param arr array para cópia.
	 */
	public void copiarDaSaida(double[] arr) {
		double[] saida = saidaParaArray();

		utils.validarNaoNulo(arr, "O array de cópia não pode ser nulo.");
		
		if (saida.length != arr.length) {
			throw new IllegalArgumentException(
				"\nIncompatibilidade de dimensões entre o array fornecido (" + arr.length + 
				") e o array gerado pela saída da última camada (" + saida.length + ")."
			);
		}

		System.arraycopy(saida, 0, arr, 0, saida.length);
	}

	/**
	 * Informa o nome configurado do modelo.
	 * @return nome do modelo.
	 */
	public String nome() {
		return this.nome;
	}

	/**
	 * Retorna a quantidade total de parâmetros do modelo.
	 * <p>
	 *    isso inclui todos os kernels e bias (caso configurados).
	 * </p>
	 * @return quantiade de parâmetros total do modelo.
	 */
	public abstract int numParametros();

	/**
	 * Retorna a quantidade de camadas presente no modelo.
	 * @return quantidade de camadas do modelo.
	 */
	public abstract int numCamadas();

	/**
	 * Disponibiliza o histórico da função de perda do modelo durante cada época
	 * de treinamento.
	 * <p>
	 *    O histórico será o do ultimo processo de treinamento usado, seja ele 
	 *    sequencial ou em lotes.
	 * </p>
	 * @return array contendo o valor de perda durante cada época de treinamento 
	 * do modelo.
	 */
	public double[] historico() {
		return _treinador.historico();
	}

	/**
	 * Gera uma string representando as características do modelo.
	 * @return {@code String} representando o modelo.
	 */
	protected abstract String construirInfo();

	/**
	 * Exibe, via console, as informações do modelo.
	 */
	public abstract void info();

	@Override
	public String toString(){
		return construirInfo();
	}

	/**
	 * Clona as características principais do modelo.
	 * @return clone do modelo.
	 */
	@Override
	public Modelo clone() {
		try {
			Modelo clone = (Modelo) super.clone(); 
			return clone;

		} catch (CloneNotSupportedException e) {
			throw new RuntimeException(e);
		}
	}
}
