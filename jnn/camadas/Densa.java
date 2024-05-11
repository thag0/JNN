package jnn.camadas;

import jnn.ativacoes.Ativacao;
import jnn.ativacoes.Linear;
import jnn.core.Dicionario;
import jnn.core.Utils;
import jnn.core.tensor.OpTensor;
import jnn.core.tensor.Tensor;
import jnn.core.tensor.Variavel;
import jnn.inicializadores.GlorotUniforme;
import jnn.inicializadores.Inicializador;
import jnn.inicializadores.Zeros;

/**
 * <h2>
 *    Camada Densa ou Totalmente Conectada
 * </h2>
 * <p>
 *    A camada densa é um tipo de camada que está profundamente conectada
 *    com a camada anterior, onde cada conexão da camada anterior se conecta
 *    com todas as conexões de saída da camada densa.
 * </p>
 * <p>
 *    Ela funciona realizando a operação de produto entre a {@code entrada} e 
 *    seus {@code pesos}, adicionando os bias caso sejam configurados, de acordo 
 *    com a expressão:
 * </p>
 * <pre>
 *    somatorio = matMult(pesos * entrada) + bias
 * </pre>
 * Após a propagação dos dados, a função de ativação da camada é aplicada ao 
 * resultado do somatório, que por fim é salvo na saída da camada.
 * <pre>
 *    saida = ativacao(somatorio)
 * </pre>
 */
public class Densa extends Camada implements Cloneable {

	/**
	 * Operador para tensores.
	 */
	private OpTensor optensor = new OpTensor();

	/**
	 * Utilitário.
	 */
	private Utils utils = new Utils();

	/**
	 * Variável controlador para o tamanho de entrada da camada densa.
	 */
	private int tamEntrada;
	 
	/**
	 * Variável controlador para a quantidade de neurônios (unidades) 
	 * da camada densa.
	 */
	private int numNeuronios;

	//core

	/**
	 * Tensor contendo os valores dos pesos de cada conexão da
	 * entrada com a saída da camada.
	 * <p>
	 *    O formato da matriz de pesos é definido por:
	 * </p>
	 * <pre>
	 *    pesos = (entrada, neuronios)
	 * </pre>
	 * Assim, a disposição dos pesos é dada da seguinte forma:
	 * <pre>
	 * pesos = [
	 *    n1p1, n2p1, n3p1, nNpN
	 *    n1p2, n2p2, n3p2, nNpN
	 *    n1p3, n2p3, n3p3, nNpN
	 * ]
	 * </pre>
	 * Onde <strong>n</strong> é o neurônio (ou unidade) e <strong>p</strong>
	 * é seu peso correspondente.
	 */
	public Tensor _pesos;

	/**
	 * Tensor contendo os viéses da camada, seu formato se dá por:
	 * <pre>
	 * b = (neuronios)
	 * </pre>
	 */
	public Tensor _bias;

	/**
	 * Auxiliar na verificação do uso do bias na camada.
	 */
	private boolean usarBias = true;

	// auxiliares

	/**
	 * Tensor contendo os valores de entrada da camada, seu formato se dá por:
	 * <pre>
	 *    entrada = (tamEntrada)
	 * </pre>
	 */
	public Tensor _entrada;

	/**
	 * Tensor contendo os valores de resultado da multiplicação matricial entre
	 * os pesos e a entrada da camada adicionados com o bias, seu formato se dá por:
	 * <pre>
	 *    somatorio = (neuronios)
	 * </pre>
	 */
	public Tensor _somatorio;

	/**
	 * Tensor contendo os valores de resultado da soma entre os valores 
	 * da matriz de somatório com os valores da matriz de bias da camada, seu 
	 * formato se dá por:
	 * <pre>
	 *    saida = (neuronios)
	 * </pre>
	 */
	public Tensor _saida;
	
	/**
	 * Tensor contendo os valores de gradientes de cada neurônio da 
	 * camada, seu formato se dá por:
	 * <pre>
	 *    gradSaida = (neuronios)
	 * </pre>
	 */
	public Tensor _gradSaida;

	/**
	 * Gradientes usados para retropropagar os erros para camadas anteriores.
	 * <p>
	 *    O formato do gradiente de entrada é definido por:
	 * </p>
	 * <pre>
	 *    gradEntrada = (tamEntrada)
	 * </pre>
	 */
	public Tensor _gradEntrada;

	/**
	 * Tensor contendo os valores dos gradientes para os pesos da camada.
	 * <p>
	 *    O formato da matriz de gradiente dos pesos é definido por:
	 * </p>
	 * <pre>
	 *    gradPesos = (entrada, neuronios)
	 * </pre>
	 */
	public Tensor _gradPesos;

	/**
	 * Tensor contendo os valores dos gradientes para os bias da camada.
	 * <p>
	 *    O formato da matriz de gradientes dos bias é definido por:
	 * </p>
	 * <pre>
	 *    gradBias = (neuronios)
	 * </pre>
	 */
	public Tensor _gradBias;

	/**
	 * Função de ativação da camada
	 */
	private Ativacao ativacao = new Linear();

	/**
	 * Inicializador para os pesos da camada.
	 */
	private Inicializador iniKernel = new GlorotUniforme();

	/**
	 * Inicializador para os bias da camada.
	 */
	private Inicializador iniBias = new Zeros();

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param e quantidade de conexões de entrada.
	 * @param n quantidade de neurônios.
	 * @param ativacao função de ativação que será usada pela camada.
	 * @param usarBias uso de viés na camada.
	 * @param iniKernel inicializador para os pesos da camada.
	 * @param iniBias inicializador para os bias da camada.
	 */
	public Densa(int e, int n, Object ativacao, Object iniKernel, Object iniBias) {
		this(n, ativacao, iniKernel, iniBias);

		if (e < 1) {
			throw new IllegalArgumentException(
				"\nA camada deve conter ao menos uma entrada."
			);
		}

		if (e <= 0) {
			throw new IllegalArgumentException(
				"\nO valor de entrada deve ser maior que zero."
			);
		}
	
		construir(new int[]{1, e});//construir automaticamente
	}

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param e quantidade de conexões de entrada.
	 * @param n quantidade de neurônios.
	 * @param ativacao função de ativação que será usada pela camada.
	 * @param iniKernel inicializador para os pesos da camada.
	 */
	public Densa(int e, int n, Object ativacao, Object iniKernel) {
		this(e, n, ativacao, iniKernel, null);
	}

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param e quantidade de conexões de entrada.
	 * @param n quantidade de neurônios.
	 * @param ativacao função de ativação que será usada pela camada.
	 */
	public Densa(int e, int n, Object ativacao) {
		this(e, n, ativacao, null, null);
	}

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param e quantidade de conexões de entrada.
	 * @param n quantidade de neurônios.
	 */
	public Densa(int e, int n) {
		this(e, n, null, null, null);
	}

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param n quantidade de neurônios.
	 * @param ativacao função de ativação que será usada pela camada.
	 * @param iniKernel inicializador para os pesos da camada.
	 * @param iniBias inicializador para os bias da camada.
	 */
	public Densa(int n, Object ativacao, Object iniKernel, Object iniBias) {
		if (n < 1) {
			throw new IllegalArgumentException(
				"A camada deve conter ao menos um neurônio."
			);
		}
		this.numNeuronios = n;

		//usar os valores padrão se necessário
		Dicionario dic = new Dicionario();
		if (ativacao != null) this.ativacao = dic.getAtivacao(ativacao);
		if (iniKernel != null) this.iniKernel = dic.getInicializador(iniKernel);
		if (iniBias != null)  this.iniBias = dic.getInicializador(iniBias);
	}

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param n quantidade de neurônios.
	 * @param ativacao função de ativação que será usada pela camada.
	 * @param iniKernel inicializador para os pesos da camada.
	 */
	public Densa(int n, Object ativacao, Object iniKernel) {
		this(n, ativacao, iniKernel, null);
	}

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param n quantidade de neurônios.
	 * @param ativacao função de ativação que será usada pela camada.
	 */
	public Densa(int n, Object ativacao) {
		this(n, ativacao, null, null);
	}

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param n quantidade de neurônios.
	 */
	public Densa(int n) {
		this(n, null, null, null);
	}

	/**
	 * Inicializa os parâmetros necessários para a camada Densa.
	 * <p>
	 *    O formato de entrada deve ser um array contendo o tamanho de 
	 *    cada dimensão e entrada da camada, e deve estar no formato:
	 * </p>
	 * <pre>
	 *    entrada = (1, tamEntrada)
	 * </pre>
	 * @param entrada formato de entrada para a camada.
	 */
	@Override
	public void construir(Object entrada) {
		if (!(entrada instanceof int[])) {
			throw new IllegalArgumentException(
				"\nObjeto esperado para entrada da camada Densa é do tipo int[], " +
				"objeto recebido é do tipo " + entrada.getClass().getTypeName()
			);
		}

		int[] formatoEntrada = (int[]) entrada;
		if (!utils.apenasMaiorZero(formatoEntrada)) {
			throw new IllegalArgumentException(
				"\nOs valores recebidos para o formato de entrada devem ser maiores que zero."
			);
		}

		this.tamEntrada = formatoEntrada[utils.ultimoIndice(formatoEntrada)];

		if (this.numNeuronios <= 0) {
			throw new IllegalArgumentException(
				"\nO número de neurônios para a camada Densa não foi definido."
			);
		}

		//inicializações
		_entrada =    new Tensor(this.tamEntrada);
		_saida =      new Tensor(this.numNeuronios);
		_pesos =      new Tensor(this.tamEntrada, this.numNeuronios);
		_gradPesos =  new Tensor(_pesos.shape());

		if (usarBias) {
			_bias =     new Tensor(_saida.shape());
			_gradBias = new Tensor(_saida.shape());
		}

		_somatorio =   new Tensor(_saida.shape());
		_gradSaida =   new Tensor(_saida.shape());
		_gradEntrada = new Tensor(_entrada.shape());

		setNomes();
		
		_treinavel = true;//camada pode ser treinada.
		_construida = true;//camada pode ser usada.
	}

	@Override
	public void setSeed(long seed) {
		iniKernel.setSeed(seed);
		iniBias.setSeed(seed);
	}

	@Override
	public void inicializar() {
		verificarConstrucao();

		iniKernel.inicializar(_pesos);

		if (usarBias) {
			iniBias.inicializar(_bias);
		}
	}

	@Override
	public void setAtivacao(Object ativacao) {
		this.ativacao = new Dicionario().getAtivacao(ativacao);
	}

	@Override
	public void setBias(boolean usarBias) {
		this.usarBias = usarBias;
	}

	@Override
	protected void setNomes() {
		_entrada.nome("entrada");
		_pesos.nome("kernel");
		_saida.nome("saida");
		_somatorio.nome("somatório");
		_gradSaida.nome("gradiente saída");
		_gradEntrada.nome("gradiente entrada");
		_gradPesos.nome("gradiente kernel");

		if (usarBias) {
			_bias.nome("bias");
			_gradBias.nome("gradiente bias");
		}
	}

	/**
	 * <h2>
	 *    Propagação direta através da camada Densa
	 * </h2>
	 * <p>
	 *    Alimenta os dados de entrada para a saída da camada por meio da 
	 *    multiplicação matricial entre a entrada recebida da camada e os pesos 
	 *    da camada, em seguida é adicionado o bias caso ele seja configurado 
	 *    no momento da inicialização.
	 * </p>
	 * <p>
	 *    A expressão que define a saída é dada por:
	 * </p>
	 * <pre>
	 *somatorio = matMult(entrada, pesos)
	 *somatorio.add(bias)
	 *saida = ativacao(somatorio)
	 * </pre>
	 * Após a propagação dos dados, a função de ativação da camada é aplicada
	 * ao resultado do somatório e o resultado é salvo da saída da camada.
	 * @param entrada dados de entrada que serão processados, objetos aceitos incluem:
	 * {@code Tensor}, ou {@code double[]}.
	 */
	@Override
	public Tensor forward(Object entrada) {
		verificarConstrucao();

		if (entrada instanceof Tensor) {
			Tensor e = (Tensor) entrada;
			if (!_entrada.compararShape(e)) {
				throw new IllegalArgumentException(
					"\nTensor esperado para camada Densa de ser " + _entrada.shapeStr() + 
					", mas possui " + e.shapeStr() + "."
				);
			}

			_entrada.copiar(e);

		} else if (entrada instanceof double[]) {
			double[] e = (double[]) entrada;
			if (e.length != _entrada.tamanho()) {
				throw new IllegalArgumentException(
					"Dimensões incompatíveis entre a entrada recebida (" + e.length +") e a" +
					" entrada da camada " + _entrada.tamanho()
				);
			}

			_entrada.copiar(e);

		} else {
			throw new IllegalArgumentException(
				"Tipo de entrada \"" + entrada.getClass().getTypeName() + "\"" +
				" não suportada."
			);
		}

		//feedforward
		_somatorio.copiar(optensor.matMult(_entrada, _pesos));

		if (usarBias) {
			_somatorio.add(_bias);
		}

		ativacao.forward(_somatorio, _saida);

		return _saida;
	}

	/**
	 * <h2>
	 *    Propagação reversa através da camada Densa
	 * </h2>
	 * <p>
	 *    Calcula os gradientes da camada para os pesos e bias baseado nos
	 *    gradientes fornecidos.
	 * </p>
	 * <p>
	 *    Após calculdos, os gradientes em relação a entrada da camada são
	 *    calculados e salvos em {@code gradEntrada} para serem retropropagados 
	 *    para as camadas anteriores do modelo em que a camada estiver.
	 * </p>
	 * Resultados calculados ficam salvos nas prorpiedades {@code camada.gradPesos} e
	 * {@code camada.gradBias}.
	 * @param grad gradiente da camada seguinte, deve ser um objeto do tipo 
	 * {@code Tensor} ou {@code double[]}.
	 */
	@Override
	public Tensor backward(Object grad) {
		verificarConstrucao();

		if (grad instanceof Tensor) {
			Tensor g = (Tensor) grad;
			if (g.numDim() != 1) {
				throw new IllegalArgumentException(
					"\nTensor de entrada deve ter 1 dimensão, mas tem " + g.numDim() + "."
				);
			}
			_gradSaida.copiar(g);

		}else if (grad instanceof double[]) {
			double[] g = (double[]) grad;
			if (g.length != _gradSaida.tamanho()) {
				throw new IllegalArgumentException(
					"\nTamanho do gradiente recebido (" + g.length + ") incompatível com o " +
					"suportado pela camada Densa (" + _gradSaida.tamanho() + ")."
				);
			}

			_gradSaida.copiar(g);
		
		}  else {
			throw new IllegalArgumentException(
				"\nO gradiente para a camada Densa deve ser do tipo " + this._gradSaida.getClass() +
				" ou \"double[]\", objeto recebido é do tipo \"" + grad.getClass().getTypeName() + "\""
			);
		}

		//backward
		ativacao.backward(this);

		_gradPesos.add(optensor.matMult(_entrada.transpor(), _gradSaida));

		if (usarBias) {
			_gradBias.add(_gradSaida);
		}

		_gradEntrada.copiar(optensor.matMult(_gradSaida, _pesos.transpor()));

		return _gradEntrada;
	}

	@Override
	public Tensor saida() {
		return _saida;
	}

	/**
	 * Retorna a quantidade de neurônios presentes na camada.
	 * @return quantidade de neurônios presentes na camada.
	 */
	public int numNeuronios() {
		verificarConstrucao();

		return _pesos.shape()[1];
	}

	@Override
	public Ativacao ativacao() {
		return ativacao;
	}

	/**
	 * Retorna a capacidade de entrada da camada.
	 * @return tamanho de entrada da camada.
	 */
	public int tamanhoEntrada() {
		verificarConstrucao();

		return _entrada.tamanho();
	}

	/**
	 * Retorna a capacidade de saída da camada.
	 * @return tamanho de saída da camada.
	 */
	public int tamanhoSaida() {
		return numNeuronios;
	}

	@Override
	public boolean temBias() {
		return usarBias;
	}

	@Override
	public int numParametros() {
		verificarConstrucao();

		int parametros = _pesos.tamanho();
		
		if (usarBias) parametros += _bias.tamanho();

		return parametros;
	}

	@Override
	public Variavel[] saidaParaArray() {
		verificarConstrucao();
		
		return _saida.paraArray();
	}

	@Override
	public String info() {
		verificarConstrucao();

		StringBuilder sb = new StringBuilder();
		String pad = " ".repeat(4);
		
		sb.append(nome() + " (id " + id + ") = [\n");

		sb.append(pad).append("Ativação: " + ativacao.nome() + "\n");
		sb.append(pad).append("Entrada: " + tamanhoEntrada() + "\n");
		sb.append(pad).append("Neurônios: " + numNeuronios() + "\n");
		sb.append(pad).append("Saida: " + tamanhoSaida() + "\n");
		sb.append("\n");

		sb.append(pad + "Pesos:  " + _pesos.shapeStr() + "\n");

		sb.append(pad + "Bias:   ");
		if (temBias()) {
			sb.append(_bias.shapeStr() + "\n");
		} else {
			sb.append(" n/a\n");
		}

		sb.append("]\n");

		return sb.toString();
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(info());
		int tamanho = sb.length();

		sb.delete(tamanho-1, tamanho);//remover ultimo "\n"    
		
		sb.append(" <hash: " + Integer.toHexString(hashCode()) + ">");
		sb.append("\n");
		
		return sb.toString();
	}

	@Override
	public Densa clone() {
		verificarConstrucao();

		Densa clone = (Densa) super.clone();

		clone.optensor = new OpTensor();
		clone.ativacao = new Dicionario().getAtivacao(this.ativacao.nome());
		clone._treinavel = this._treinavel;

		clone.usarBias = this.usarBias;
		if (this.usarBias) {
			clone._bias = this._bias.clone();
			clone._gradBias = this._gradBias.clone();
		}

		clone._entrada = this._entrada.clone();
		clone._pesos = this._pesos.clone();
		clone._somatorio = this._somatorio.clone();
		clone._saida = this._saida.clone();
		clone._gradSaida = this._gradSaida.clone();
		clone._gradPesos = this._gradPesos.clone();

		return clone;
	}

	/**
	 * Calcula o formato de entrada da camada Densa, que é disposto da
	 * seguinte forma:
	 * <pre>
	 *    formato = (altura, largura)
	 * </pre>
	 * @return formato de entrada da camada.
	 */
	@Override
	public int[] formatoEntrada() {
		return _entrada.shape();
	}

	/**
	 * Calcula o formato de saída da camada Densa, que é disposto da
	 * seguinte forma:
	 * <pre>
	 *    formato = (saida.altura, saida.largura)
	 * </pre>
	 * No caso da camada densa, o formato também pode ser descrito como:
	 * <pre>
	 *    formato = (1, numNeuronios)
	 * </pre>
	 * @return formato de saída da camada
	 */
	@Override
	public int[] formatoSaida() {
		return _saida.shape();
	}

	@Override
	public Tensor kernel() {
		return _pesos;
	}

	@Override
	public Variavel[] kernelParaArray() {
		return kernel().paraArray();
	}

	@Override
	public Tensor gradKernel() {
		return _gradPesos;
	}

	@Override
	public Variavel[] gradKernelParaArray() {
		return gradKernel().paraArray();
	}

	@Override
	public Tensor bias() {
		if (usarBias) {
			return _bias;
		}

		throw new IllegalStateException(
			"\nA camada " + nome() + " (" + id + ") não possui bias configurado."
		);
	}

	@Override
	public Variavel[] biasParaArray() {
		return bias().paraArray();
	}

	@Override
	public Tensor gradBias() {
		return _gradBias;
	}

	@Override
	public Variavel[] gradBiasParaArray() {
		return gradBias().paraArray();
	}

	@Override
	public Tensor gradEntrada() {
		return _gradEntrada;
	}

	@Override
	public void setGradienteKernel(Variavel[] grads) {
		if (grads.length != _gradPesos.tamanho()) {
			throw new IllegalArgumentException(
				"A dimensão dos gradientes fornecidos não é igual a quantidade de " +
				"parâmetros para os kernels da camada (" + _gradPesos.tamanho() + ")."
			);         
		}

		int cont = 0, lin = _gradPesos.shape()[0], col = _gradPesos.shape()[1];
		for (int i = 0; i < lin; i++) {
			for (int j = 0; j < col; j++) {
				_gradPesos.set(grads[cont++].get(), 0, 0, i, j);
			}
		}
	}

	@Override
	public void setGradienteBias(Variavel[] grads) {
		_gradBias.copiarElementos(grads);
	}

	@Override
	public void setKernel(Variavel[] kernel) {
		if (kernel.length != _pesos.tamanho()) {
			throw new IllegalArgumentException(
				"A dimensão do kernel fornecido não é igual a quantidade de " +
				"parâmetros para os kernels da camada."
			);         
		}

		int cont = 0;
		int lin = _pesos.shape()[0];
		int col = _pesos.shape()[1];
		for (int i = 0; i < lin; i++) {
			for (int j = 0; j < col; j++) {
				_pesos.set(kernel[cont++].get(), i, j);
			}
		}
	}

	@Override
	public void setBias(Variavel[] bias) {
		if (bias.length != (_bias.tamanho())) {
			throw new IllegalArgumentException(
				"A dimensão do bias fornecido (" + bias.length + ") não é igual a quantidade de " +
				" parâmetros para os bias da camada (" + _bias.tamanho() + ")."
			);
		}

		_bias.copiarElementos(bias);
	}

	@Override
	public void zerarGradientes() {
		verificarConstrucao();

		_gradPesos.zerar();
		_gradBias.zerar();
	}
}
