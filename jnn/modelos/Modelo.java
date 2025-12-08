package jnn.modelos;

import jnn.camadas.Camada;
import jnn.core.Utils;
import jnn.core.tensor.Tensor;
import jnn.dataloader.DataLoader;
import jnn.metrica.Avaliador;
import jnn.metrica.perda.Perda;
import jnn.otm.Otimizador;
import jnn.treino.Treinador;

/**
 * <h2>
 *    Modelo base
 * </h2>
 * Inteface para modelos criados dentro da biblioteca.
 */
public abstract class Modelo implements Cloneable, Iterable<Camada> {

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
	 *		Uma nova seed só é configurada se seu valor for
	 *		diferente de zero.
	 * </p>
	 */
	protected long seedInicial = 0;

	/**
	 * Gerenciador de treino do modelo.
	 */
	protected Treinador _treinador;

	/**
	 * Auxiliar na verificação de armazenagem do histórico
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
	protected Utils utils;
	
	/**
	 * Auxiliar de verificação da alteração do método de treino.
	 */
	protected boolean configTreino = false;

	/**
	 * Inicialização implicita de um modelo.
	 */
	protected Modelo() {
		_treinador = new Treinador(this);
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
	public void setSeed(Number seed) {
		utils.validarNaoNulo(seed, "Seed nula.");
		seedInicial = seed.longValue();
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
	 * @param loss nova função de perda.
	 */
	public void setPerda(Perda loss) {
		utils.validarNaoNulo(loss, "loss == null.");
		_perda = loss;
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
	 * @param otm novo otimizador.
	 */
	public void setOtimizador(Otimizador otm) {
		utils.validarNaoNulo(otm, "otm == null.");
		_otimizador = otm;
	}

	/**
	 * Configura um novo treinador para o modelo.
	 * @param t {@code Treinador} novo.
	 */
	public void setTreinador(Treinador t) {
		utils.validarNaoNulo(t, "t == null.");
		_treinador = t;
		configTreino = true;
	}

	/**
	 * Inicializa os parâmetros necessários para a criação do modelo,
	 * além de gerar os valores iniciais para os kernels e bias.
	 * <p>
	 *    Caso nenhuma configuração inicial seja feita, o modelo será 
	 *    compilado com os valores padrões. 
	 * </p>
	 * <p>
	 *    Otimizadores podem ser recebidos usando instâncias pré configuradas, 
	 *    essas intâncias dão a liberdade de inicializar o otimizador com valores
	 *    personalizáveis para seus parâmetros (como taxa de aprenziado, por exemplo).
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
	public abstract void compilar(Object otm, Object loss);

	/**
	 * Auxiliar na verificação da compilação do modelo.
	 */
	protected void validarCompilacao() {
		if (!_compilado) {
			throw new IllegalStateException(
				"\nO modelo deve ser compilado."
			);
		}
	}

	/**
	 * Propaga os dados de entrada através das camadas do modelo.
	 * @param x {@code Tensor} de entrada.
	 * @return {@code Tensor} contendo a saída prevista pelo modelo.
	 */
	public Tensor forward(Tensor x) {
		validarCompilacao();

		for (Camada camada : this) {
			x = camada.forward(x);
		}

		return x;
	}

	/**
	 * Alimenta o modelo com vários dados de entrada.
	 * @param xs array de {@code Tensor} contendo múltiplas 
	 * entradas para o modelo.
	 * @return array de {@code Tensor} contendo as previsões 
	 * correspondentes.
	 */
	public Tensor[] forward(Tensor[] xs) {
		validarCompilacao();

		utils.validarNaoNulo(xs, "xs == null.");
		
		Tensor y = forward(utils.concatenar(xs));
		
		Tensor[] prevs = new Tensor[xs.length];
		for (int i = 0; i < xs.length; i++) {
			prevs[i] = new Tensor(y.subTensor(i));
		}

		return prevs;
	}

	/**
	 * Realiza a propagação reversa do gradiente através do modelo.
	 * @param g {@code Tensor} contendo o gradiente da perda em relação a saída do modelo.
	 * @return {@code Tensor} contendo o gradiente da perda em relação a entrada do modelo.
	 */
	public abstract Tensor backward(Tensor g);

	/**
	 * Zera os gradientes acumulados do modelo.
	 * <p>
	 *    Apenas camadas treináveis são afetadas.
	 * </p>
	 */
	public void gradZero() {
		for (Camada camada : this) {
			if (camada.treinavel()) camada.gradZero();
		}
	}

	/**
	 * Realiza as verificações necessárias nos dados usados pelo modelo.
	 * @param <T> tipo de dados, comumente usando {@code Tensor}.
	 * @param xs array contendos dados de entrada.
	 * @param ys array contendos dados de saída.
	 */
	private <T> void validarDados(T[] xs, T[] ys) {
		utils.validarNaoNulo(xs, "xs == null.");
		utils.validarNaoNulo(ys, "ys == null.");
 
		if (xs.length != ys.length) {
			throw new IllegalArgumentException(
				"\nDados de entrada e saída devem conter o mesmo tamanho, " +
				"recebido x = " + xs.length + " e y = " + ys.length
			);
		}
	}

	/**
	 * Treina o modelo de acordo com as configurações predefinidas.
	 * @param xs dados de entrada do treino (features).
	 * @param ys dados de saída correspondente a entrada (classes).
	 * @param epochs quantidade de épocas de treinamento.
	 * @param logs logs para perda durante as épocas de treinamento.
	 */
	public void treinar(Tensor[] xs, Tensor[] ys, int epochs, boolean logs) {
		treinar(xs, ys, epochs, 1, logs);
	}
	
	/**
	 * Treina o modelo de acordo com as configurações predefinidas utilizando o
	 * treinamento em lotes.
	 * @param xs {@code Tensores} contendos os dados de entrada.
	 * @param ys {@code Tensores} contendos os dados de saída (rótulos).
	 * @param epochs quantidade de épocas de treinamento.
	 * @param tamLote tamanho do lote de treinamento.
	 * @param logs logs para perda durante as épocas de treinamento.
	 */
	public void treinar(Tensor[] xs, Tensor[] ys, int epochs, int tamLote, boolean logs) {
		validarCompilacao();
		validarDados(xs, ys);

		if (epochs < 1) {
			throw new IllegalArgumentException(
				"\nValor de épocas deve ser maior que zero, recebido = " + epochs
			);
		}

		if (tamLote < 1) {
			throw new IllegalArgumentException(
				"\nValor de lote deve ser maior que zero, recebido = " + tamLote
			);
		}
		
		_treinador.executar(xs, ys, epochs, tamLote, logs);
	}
	
	/**
	 * Treina o modelo de acordo com as configurações predefinidas.
	 * @param loader {@code DataLoader} com dados de treino.
	 * @param epochs quantidade de épocas de treinamento.
	 * @param logs logs para perda durante as épocas de treinamento.
	 */
	public void treinar(DataLoader loader, int epochs, boolean logs) {
		treinar(loader, epochs, 1, logs);
	}

	/**
	 * Treina o modelo de acordo com as configurações predefinidas
	 * utilizando o treinamento em lotes.
	 * @param loader {@code DataLoader} com dados de treino.
	 * @param epochs quantidade de épocas de treinamento.
	 * @param tamLote tamanho do lote de treinamento.
	 * @param logs logs para perda durante as épocas de treinamento.
	 */
	public void treinar(DataLoader loader, int epochs, int tamLote, boolean logs) {
		validarCompilacao();

		if (epochs < 1) {
			throw new IllegalArgumentException(
				"\nValor de épocas deve ser maior que zero, recebido = " + epochs
			);
		}

		if (tamLote < 1) {
			throw new IllegalArgumentException(
				"\nValor de lote deve ser maior que zero, recebido = " + tamLote
			);
		}
		
		_treinador.executar(loader, epochs, tamLote, logs);
	}

	/**
	 * Configura o modelo para o modo de treino
	 * @param is {@code true} caso o modelo deva estar no modo treino,
	 * {@code false} caso contrário.
	 */
	public void treino(boolean is) {
		for (Camada camada : this) {
			camada.setTreino(is);
		}
	}

	/**
	 * Avalia o modelo, utilizando a função de perda configurada.
	 * <p>
	 *    É possível utilizar outras funções de perda mesmo que sejam diferentes
	 *    da que o modelo usa, através de:
	 * </p>
	 * <pre>
	 * modelo.avaliador()
	 * </pre>
	 * @param xs {@code Tensores} contendo dados de entrada.
	 * @param ys {@code Tensores} contendo dados de saída correspondente as entradas fornecidas.
	 * @return {@code Tensor} contendo valor de perda.
	 */
	public Tensor avaliar(Tensor[] xs, Tensor[] ys) {
		validarCompilacao();
		validarDados(xs, ys);

		// esse tamamho foi o mais ou menos ideial pra não ficar usando
		// tanta memória e teve um leve ganho em tempo de execução.
		final int batch = 32;

		final int n = xs.length;
		double loss = 0;
		for (int i = 0; i < n; i += batch) {
			int inicio = i;
			int fim = Math.min(inicio + batch, n);
			Tensor x = utils.concatenar(utils.subArray(xs, inicio, fim));
			Tensor y = utils.concatenar(utils.subArray(ys, inicio, fim));
			
			Tensor prev = forward(x);

			double lossLote = _perda.forward(prev, y).item();
			int tamLote = fim - inicio;
			loss += lossLote * tamLote;
		}

		return new Tensor(
			new double[] { loss / n }
		);
	}

	/**
	 * Avalia o modelo, utilizando a função de perda configurada.
	 * <p>
	 *    É possível utilizar outras funções de perda mesmo que sejam diferentes
	 *    da que o modelo usa, através de:
	 * </p>
	 * <pre>
	 * modelo.avaliador()
	 * </pre>
	 * @param loader {@code DataLoader} contendo dados de avaliação.
	 * @return {@code Tensor} contendo valor de perda.
	 */
	public Tensor avaliar(DataLoader loader) {
		return avaliar(loader.getX(), loader.getY());
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
		return _avaliador;
	}

	/**
	 * Retorna o treinador do modelo.
	 * @return treinador do modelo.
	 */
	public Treinador treinador() {
		return _treinador;
	}

	/**
	 * Retorna o otimizador configurado para o treino do modelo modelo.
	 * @return otimizador atual do modelo.
	 */
	public Otimizador otm() {
		validarCompilacao();
		return _otimizador;
	}

	/**
	 * Retorna a função de perda configurada do modelo.
	 * @return função de perda atual do modelo.
	 */
	public Perda loss() {
		validarCompilacao();
		return _perda;
	}

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
	 * Retorna o conjunto de parâmetros do modelo.
	 * <p>
	 *		Os parâmetros de um modelo incluem {@code kernels} e {@code bias}
	 *		de cada camada, todos sendo do tipo {@code Tensor}.
	 * </p>
	 * <p>
	 * 		A sequência fornecida dos parâmetros é dada por:
	 * </p>
	 * <pre>
	 * 	params = [k1, b1, k2, b2, k3, b3, ...]
	 * </pre>
	 * Onde: {@code k = kernel} e {@code b = bias}.
	 * @return array de {@code Tensor} contendo os parâmetros do modelo.
	 */
	public Tensor[] params() {
		Tensor[] params = new Tensor[0];

		for (Camada camada : this) {
			if (camada.treinavel()) {
				params = utils.addEmArray(params, camada.kernel());
				if (camada.temBias()) {
					params = utils.addEmArray(params, camada.bias());
				}
			}
		}

		return params;
	}

	/**
	 * Retorna o conjunto de gradientes em relação aos parâmetros do modelo.
	 * <p>
	 * 		A sequência fornecida dos gradientes é dada por:
	 * </p>
	 * <pre>
	 * 	params = [gk1, gb1, gk2, gb2, gk3, gb3, ...]
	 * </pre>
	 * Onde: {@code gk = gradKernel} e {@code gb = gradBias}.
	 * @return array de {@code Tensor} contendo os gradientes do modelo.
	 */
	public Tensor[] grads() {
		Tensor[] grads = {};

		for (Camada camada : this) {
			if (camada.treinavel()) {
				grads = utils.addEmArray(grads, camada.gradKernel());
				if (camada.temBias()) {
					grads = utils.addEmArray(grads, camada.gradBias());
				}
			}
		}

		return grads;
	}
	
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
		utils.validarNaoNulo(arr, "arr == null.");
		
		double[] saida = saidaParaArray();
		
		if (saida.length != arr.length) {
			throw new IllegalArgumentException(
				"\nIncompatibilidade de dimensões entre o array fornecido (" + arr.length + 
				") e o array gerado pela saída da última camada (" + saida.length + ")."
			);
		}

		System.arraycopy(saida, 0, arr, 0, arr.length);
	}

	/**
	 * Informa o nome configurado do modelo.
	 * @return nome do modelo.
	 */
	public String nome() {
		return nome;
	}

	/**
	 * Retorna a quantidade total de parâmetros do modelo.
	 * <p>
	 *    isso inclui todos os kernels e bias (caso configurados).
	 * </p>
	 * @return quantidade de parâmetros total do modelo.
	 */
	public int numParams() {
		int params = 0;

		for (Camada camada : this) {
			params += camada.numParams();
		}

		return params;
	}

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
	public double[] hist() {
		return _treinador.hist();
	}

	/**
	 * Gera uma string representando as características do modelo.
	 * @return {@code String} representando o modelo.
	 */
	protected abstract String construirInfo();

	/**
	 * Exibe, via console, as informações do modelo.
	 */
	public abstract void print();

	@Override
	public String toString() {
		validarCompilacao();
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
