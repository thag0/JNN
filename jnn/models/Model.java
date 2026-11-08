package jnn.models;

import jnn.core.JNNutils;
import jnn.core.Parametro;
import jnn.core.tensor.Tensor;
import jnn.dataloader.DataLoader;
import jnn.layers.Layer;
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
public abstract class Model implements Cloneable, Iterable<Layer> {

	/**
	 * Nome da instância do modelo.
	 */
	protected String name = getClass().getSimpleName();

	/**
	 * <p>
	 *		Auxiliar no controle da compilação do modelo, ajuda a 
	 *		evitar uso indevido caso ainda não tenha suas variáveis 
	 *		e dependências inicializadas previamente.
	 * </p>
	 */
	public boolean _compiled;//mudar isso para protected assim que possivel

	/**
	 * Função de perda para avaliar o erro durante o treino.
	 */
	protected Perda _loss;

	/**
	 * Otimizador usado para ajuste de parâmetros treináveis.
	 */
	protected Otimizador _optimizer;

	/**
	 * Gerenciador de treino do modelo.
	 */
	protected Treinador _trainer;

	/**
	 * Auxiliar na verificação de armazenagem do histórico
	 * de perda do modelo durante o treinamento.
	 */
	protected boolean trainHistory = false;

	/**
	 * Responsável pelo retorno de desempenho do modelo.
	 * Contém implementações de métodos tanto para cálculo de perdas
	 * quanto de métricas.
	 * <p>
	 *    Cada modelo possui seu próprio avaliador.
	 * </p>
	 */
	protected Avaliador _evaluator;

	/**
	 * Inicialização implicita de um modelo.
	 */
	protected Model() {
		_trainer = new Treinador(this);
		_evaluator = new Avaliador(this);
	}

	/**
	 * <p>
	 *    Altera o nome do modelo.
	 * </p>
	 * O nome é apenas estético e não influencia na performance ou na 
	 * usabilidade do modelo.
	 * @param name novo nome da rede.
	 */
	public void setName(String name) {
		if (name != null) {
			String s = name.trim();
			if (!s.isEmpty()) this.name = s;
		}
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
	 * @param save se verdadeiro, o modelo armazenará o histórico de perda 
	 * durante cada época de treinamento.
	 */
	public void setHistorico(boolean save) {
		trainHistory = save;
		_trainer.setHistorico(save);
	}

	/**
	 * Configura a função de perda que será utilizada durante o processo
	 * de treinamento do modelo.
	 * @param loss nova função de perda.
	 */
	public void setLoss(Perda loss) {
		_loss = JNNutils.validarNaoNulo(loss, "loss == null.");
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
	 * @param opt novo otimizador.
	 */
	public void setOptimizer(Otimizador opt) {
		_optimizer = JNNutils.validarNaoNulo(opt, "otm == null.");
	}

	/**
	 * Inicializa os parâmetros necessários para a criação do modelo.
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
	 * @param opt otimizador usando para ajustar os parâmetros treinavéis do 
	 * modelo, pode ser uma {@code String} referente ao nome ou uma {@code instância} 
	 * já inicializada.
	 * @param loss função de perda usada para avaliar o erro do modelo durante o 
	 * treino, pode ser uma {@code String} referente ao nome ou uma {@code instância} 
	 * já inicializada.
	 */
	public abstract void compile(Object opt, Object loss);

	/**
	 * Auxiliar na verificação da compilação do modelo.
	 */
	protected void validateCompilation() {
		if (!_compiled) {
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
		validateCompilation();

		for (Layer layer : this) {
			x = layer.forward(x);
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
		validateCompilation();

		JNNutils.validarNaoNulo(xs, "xs == null.");
		
		Tensor y = forward(JNNutils.concatenar(xs));
		
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
	public Tensor backward(Tensor g) {
		validateCompilation();
		
		try {
			final int n = numLayers() - 1;
			for (int i = n; i >= 0; i--) {
				g = layer(i).backward(g);
			}
		
		} catch (NullPointerException npe) {
			// Pode disparar um nullpointer caso não tenha rodado
			// nenhum forward anteriormente, aí as entradas das
			// camadas ainda não apontam para lugar nenhum.
			throw new IllegalStateException(
				"\nNecessário realizar um forward prévio antes de chamar backward()."
			);
		}

		return g;
	}

	/**
	 * Zera os gradientes acumulados do modelo.
	 * <p>
	 *    Apenas camadas treináveis são afetadas.
	 * </p>
	 */
	public void zeroGrad() {
		for (Layer layer : this) {
			if (layer.isTreinable()) layer.zeroGrad();
		}
	}
	
	/**
	 * Treina o modelo de acordo com as configurações predefinidas utilizando o
	 * treinamento em lotes.
	 * @param loader conjunto de dados para treino.
	 * @param epochs quantidade de épocas de treinamento.
	 * @param batchSize tamanho do lote de treinamento.
	 * @param logs logs para perda durante as épocas de treinamento.
	 */
	public void fit(DataLoader loader, int epochs, int batchSize, boolean logs) {
		validateCompilation();

		if (epochs < 1) {
			throw new IllegalArgumentException(
				"\nValor de épocas deve ser maior que zero, recebido = " + epochs
			);
		}

		if (batchSize < 1) {
			throw new IllegalArgumentException(
				"\nValor de lote deve ser maior que zero, recebido = " + batchSize
			);
		}
		
		_trainer.executar(loader, epochs, batchSize, logs);
	}
	
	/**
	 * Treina o modelo de acordo com as configurações predefinidas.
	 * @param loader conjunto de dados para treino.
	 * @param epochs quantidade de épocas de treinamento.
	 * @param logs logs para perda durante as épocas de treinamento.
	 */
	public void fit(DataLoader loader, int epochs, boolean logs) {
		fit(loader, epochs, 1, logs);
	}

	/**
	 * Configura o modelo para o modo de treino
	 * @param is {@code true} caso o modelo deva estar no modo treino,
	 * {@code false} caso contrário.
	 */
	public void setTraining(boolean is) {
		for (Layer layer : this) {
			layer.setTraining(is);
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
	 * @param loader {@code DataLoader} com conjunto de dados.
	 * @return {@code Tensor} contendo valor de perda.
	 */
	public Tensor evaluate(DataLoader loader) {
		validateCompilation();
		setTraining(false);// garantia

		Tensor[] xs = loader.getX();
		Tensor[] ys = loader.getY();

		// esse tamamho foi o mais ou menos ideal pra não ficar usando
		// tanta memória e teve um leve ganho em tempo de execução.
		final int batch = 32;

		final int n = loader.len();
		float loss = 0;
		for (int i = 0; i < n; i += batch) {
			int inicio = i;
			int fim = Math.min(inicio + batch, n);
			Tensor x = JNNutils.concatenar(JNNutils.subArray(xs, inicio, fim));
			Tensor y = JNNutils.concatenar(JNNutils.subArray(ys, inicio, fim));
			
			Tensor prev = forward(x);

			float lossLote = _loss.forward(prev, y).item();
			int tamLote = fim - inicio;
			loss += lossLote * tamLote;
		}

		return new Tensor(1).set((loss / n), 0);
	}

	/**
	 * Retorna o avaliador do modelo, 
	 * <p>
	 *    O avaliador contém diferentes métodos de métricas úteis
	 *    para medir seu desempenho.
	 * </p>
	 * @return avaliador do modelo.
	 */
	public Avaliador evaluator() {
		return _evaluator;
	}

	/**
	 * Retorna o treinador do modelo.
	 * @return treinador do modelo.
	 */
	public Treinador trainer() {
		return _trainer;
	}

	/**
	 * Retorna o otimizador configurado para o treino do modelo modelo.
	 * @return otimizador atual do modelo.
	 */
	public Otimizador optim() {
		validateCompilation();
		return _optimizer;
	}

	/**
	 * Retorna a função de perda configurada do modelo.
	 * @return função de perda atual do modelo.
	 */
	public Perda loss() {
		validateCompilation();
		return _loss;
	}

	/**
	 * Retorna a {@code camada} do Modelo correspondente ao índice fornecido.
	 * @param i índice da busca.
	 * @return camada baseada na busca.
	 */
	public abstract Layer layer(int i);

	/**
	 * Retorna todo o conjunto de camadas presente no modelo.
	 * @return conjunto de camadas do modelo.
	 */
	public abstract Layer[] layers();

	/**
	 * Retorna a {@code camada de saída}, ou última camada, do modelo.
	 * @return camada de saída.
	 */
	public abstract Layer outputLayer();

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
	public Parametro[] params() {
		Parametro[] params = {};

		for (Layer layer : this) {
			if (layer.isTreinable()) {
				for (var p : layer.params()) {
					params = JNNutils.addEmArray(params, p);
				}
			}
		}

		return params;
	}

	/**
	 * Remove as dimensões de lote das camadas do modelo.
	 */
	public void zeroBatch() {
		validateCompilation();
		
		for (Layer camada : this) {
			camada.fitToBatch(0);
		}
	}
	
	/**
	 * Retorna um array contendo a saída serializada do modelo.
	 * @return saída do modelo.
	 */
	public abstract float[] saidaParaArray();

	/**
	 * Copia os dados de saída da última camada do modelo para o array.
	 * @param arr array para cópia.
	 */
	public void copyFromOutput(float[] arr) {
		JNNutils.validarNaoNulo(arr, "arr == null.");
		
		float[] saida = saidaParaArray();
		
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
	public String name() {
		return name;
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

		for (Layer camada : this) {
			params += camada.numParams();
		}

		return params;
	}

	/**
	 * Retorna a quantidade de camadas presente no modelo.
	 * @return quantidade de camadas do modelo.
	 */
	public abstract int numLayers();

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
	public float[] hist() {
		return _trainer.hist();
	}

	/**
	 * Gera uma string representando as características do modelo.
	 * @return {@code String} representando o modelo.
	 */
	protected abstract String buildInfo();

	/**
	 * Exibe, via console, as informações do modelo.
	 */
	public abstract void print();

	@Override
	public String toString() {
		validateCompilation();
		return buildInfo();
	}

	/**
	 * Clona as características principais do modelo.
	 * @return clone do modelo.
	 */
	@Override
	public Model clone() {
		try {
			Model clone = (Model) super.clone(); 
			return clone;

		} catch (CloneNotSupportedException e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * Retorna o tamanho em bytes na memória que o modelo possui alocado.
	 * @return tamanho calculado em bytes.
	 */
	public abstract long sizeBytes();

	/**
	 * Retorna o formato de entrada da primeira camada do modelo.
	 * @return array contendo formato de entrada do modelo.
	 */
	public int[] shapeIn() {
		return layer(0).shapeIn();
	}
	
	/**
	 * Retorna o formato de saída da última camada do modelo.
	 * @return array contendo formato de saída do modelo.
	 */
	public int[] shapeOut() {
		return outputLayer().shapeOut();
	}

}
