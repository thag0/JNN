package jnn.camadas;

import java.util.Optional;

import jnn.ativacoes.Ativacao;
import jnn.ativacoes.Linear;
import jnn.core.Dicionario;
import jnn.core.OpTensor;
import jnn.core.Utils;
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
 *buffer = matmul(entrada, kernel);
 *buffer.add(bias);
 * </pre>
 * Após a propagação dos dados, a função de ativação da camada é aplicada ao 
 * resultado do buffer, que por fim é salvo na saída da camada.
 * <pre>
 *    saida = ativacao(buffer);
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
	public Tensor _kernel;

	/**
	 * Tensor contendo os viéses da camada, seu formato se dá por:
	 * <pre>
	 * b = (neuronios)
	 * </pre>
	 */
	public Optional<Tensor> _bias;

	/**
	 * Auxiliar na verificação do uso do bias na camada.
	 */
	private boolean usarBias = true;

	/**
	 * Tensor contendo os valores de entrada da camada, seu formato se dá por:
	 * <pre>
	 *    entrada = (tamEntrada)
	 * </pre>
	 */
	public Tensor _entrada;

	/**
	 * Tensor contendo os valores de resultados intermediários 
	 * do processamento da camada.
	 * <pre>
	 *    buffer = (neuronios)
	 * </pre>
	 */
	public Tensor _buffer;

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
	public Tensor _gradKernel;

	/**
	 * Tensor contendo os valores dos gradientes para os bias da camada.
	 * <p>
	 *    O formato da matriz de gradientes dos bias é definido por:
	 * </p>
	 * <pre>
	 *    gradBias = (neuronios)
	 * </pre>
	 */
	public Optional<Tensor> _gradBias;

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
	 * @param n quantidade de neurônios (unidades).
	 * @param act função de ativação que será usada pela camada.
	 * @param usarBias uso de viés na camada.
	 * @param iniKernel inicializador para os pesos da camada.
	 * @param iniBias inicializador para os bias da camada.
	 */
	public Densa(int e, int n, Object act, Object iniKernel, Object iniBias) {
		this(n, act, iniKernel, iniBias);
		construir(new int[]{ e });// construir automaticamente
	}

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param e quantidade de conexões de entrada.
	 * @param n quantidade de neurônios (unidades).
	 * @param act função de ativação que será usada pela camada.
	 * @param iniKernel inicializador para os pesos da camada.
	 */
	public Densa(int e, int n, Object act, Object iniKernel) {
		this(e, n, act, iniKernel, null);
	}

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param e quantidade de conexões de entrada.
	 * @param n quantidade de neurônios (unidades).
	 * @param act função de ativação que será usada pela camada.
	 */
	public Densa(int e, int n, Object act) {
		this(e, n, act, null, null);
	}

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param e quantidade de conexões de entrada.
	 * @param n quantidade de neurônios (unidades).
	 */
	public Densa(int e, int n) {
		this(e, n, null, null, null);
	}

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param n quantidade de neurônios (unidades).
	 * @param act função de ativação que será usada pela camada.
	 * @param iniKernel inicializador para os pesos da camada.
	 * @param iniBias inicializador para os bias da camada.
	 */
	public Densa(int n, Object act, Object iniKernel, Object iniBias) {
		if (n < 1) {
			throw new IllegalArgumentException(
				"\nA camada deve conter ao menos um neurônio."
			);
		}

		numNeuronios = n;

		//usar os valores padrão se necessário
		Dicionario dic = new Dicionario();
		if (act != null) this.ativacao = dic.getAtivacao(act);
		if (iniKernel != null) this.iniKernel = dic.getInicializador(iniKernel);
		if (iniBias != null)  this.iniBias = dic.getInicializador(iniBias);
	}

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param n quantidade de neurônios (unidades).
	 * @param act função de ativação que será usada pela camada.
	 * @param iniKernel inicializador para os pesos da camada.
	 */
	public Densa(int n, Object act, Object iniKernel) {
		this(n, act, iniKernel, null);
	}

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param n quantidade de neurônios (unidades).
	 * @param act função de ativação que será usada pela camada.
	 */
	public Densa(int n, Object act) {
		this(n, act, null, null);
	}

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param n quantidade de neurônios (unidades).
	 */
	public Densa(int n) {
		this(n, null, null, null);
	}

	/**
	 * Inicializa os parâmetros necessários para a camada Densa.
	 * <p>
	 *		O formato de entrada deve ser um array contendo um elemento
	 * 		especificando a capacidade de entrada da camada.
	 * </p>
	 * Exemplo:
	 * <pre>
	 *int[] shape = {128};
	 *camada.construir(shape);
	 * </pre>
	 */
	@Override
	public void construir(int[] shape) {
		utils.validarNaoNulo(shape, "Formato de entrada nulo.");

		if (shape.length != 1) {
			throw new IllegalArgumentException(
				"\nFormato de entrada deve conter 1 elemento, recebido " + shape.length
			);
		}

		tamEntrada = shape[0];

		if (tamEntrada < 1) {
			throw new IllegalArgumentException(
				"\nTamanho de entrada deve ser maior que zero."
			);
		}

		if (numNeuronios < 1) {
			throw new IllegalArgumentException(
				"\nNúmero de neurônios para a camada Densa não foi definido."
			);
		}

		_entrada  	= addParam("Entrada", tamEntrada);
		_saida  	= addParam("Saida", numNeuronios);
		_kernel 	= addParam("Kernel", tamEntrada, numNeuronios);
		_gradKernel = addParam("Grad Kernel", _kernel.shape());

		if (usarBias) {
			_bias 	  = Optional.of(addParam("Bias", _saida.shape()));
			_gradBias = Optional.of(addParam("Grad Bias", _saida.shape()));
		}

		_buffer 	 = addParam("Buffer", _saida.shape());
		_gradSaida 	 = addParam("Grad Saida", _saida.shape());
		_gradEntrada = addParam("Grad Entrada", _entrada.shape());
		
		_treinavel = true;// camada pode ser treinada.
		_construida = true;// camada pode ser usada.
	}

	@Override
	public void setSeed(Number seed) {
		iniKernel.setSeed(seed);
		iniBias.setSeed(seed);
	}

	@Override
	public void inicializar() {
		verificarConstrucao();

		iniKernel.forward(_kernel);
		_bias.ifPresent(b -> iniBias.forward(b));
	}

	@Override
	public void setAtivacao(Object act) {
		this.ativacao = new Dicionario().getAtivacao(act);
	}

	@Override
	public void setBias(boolean usarBias) {
		this.usarBias = usarBias;
	}

	/**
	 * <h2>
	 *		Propagação direta através da camada Densa
	 * </h2>
	 * <p>
	 *		Alimenta os dados de entrada para a saída da camada por meio da 
	 *		multiplicação matricial entre a entrada recebida e os pesos da 
	 *		camada, em seguida é adicionado o bias caso ele seja configurado 
	 *		no momento da inicialização.
	 * </p>
	 * <p>
	 *		Após a propagação dos dados, a função de ativação da camada é aplicada
	 *		ao resultado, que então é salvo da saída da camada.
	 * </p>
	 * <p>
	 *    A expressão que define a saída é dada por:
	 * </p>
	 * <pre>
	 *buffer = matMult(entrada, pesos);
	 *buffer.add(bias);
	 *saida = ativacao(buffer);
	 * </pre>
	 */
	@Override
	public Tensor forward(Tensor x) {
		verificarConstrucao();

		_entrada.copiar(x);

		_buffer.zero();
		optensor.forwardDensa(_entrada, _kernel, _bias, _buffer);

		ativacao.forward(_buffer, _saida);

		return _saida;
	}

	/**
	 * <h2>
	 *		Propagação reversa através da camada Densa
	 * </h2>
	 * <p>
	 *		Calcula os gradientes da camada para os pesos e bias baseado nos
	 *		gradientes fornecidos.
	 * </p>
	 * <p>
	 *		Após calculdos, os gradientes em relação a entrada da camada são
	 *		calculados e salvos em {@code gradEntrada} para serem retropropagados 
	 *		para as camadas anteriores do modelo em que a camada estiver.
	 * </p>
	 */
	@Override
	public Tensor backward(Tensor g) {
		verificarConstrucao();

		_gradSaida.copiar(g);
		
		ativacao.backward(this);
		
		_gradEntrada.zero();
		optensor.backwardDensa(
			_entrada,
			_kernel,
			_gradSaida,
			_gradKernel,
			_gradBias,
			_gradEntrada
		);

		return _gradEntrada;
	}

	@Override
	public void gradZero() {
		verificarConstrucao();

		_gradKernel.zero();
		_gradBias.ifPresent(gb -> gb.zero());
	}

	@Override
	public Tensor saida() {
		verificarConstrucao();
		return _saida;
	}

	/**
	 * Retorna a quantidade de neurônios presentes na camada.
	 * @return quantidade de neurônios presentes na camada.
	 */
	public int numNeuronios() {
		verificarConstrucao();
		return _kernel.shape()[1];
	}

	@Override
	public Ativacao ativacao() {
		return ativacao;
	}

	/**
	 * Retorna a capacidade de entrada da camada.
	 * @return tamanho de entrada da camada.
	 */
	public int tamEntrada() {
		verificarConstrucao();
		return _entrada.tam();
	}

	/**
	 * Retorna a capacidade de saída da camada.
	 * @return tamanho de saída da camada.
	 */
	public int tamSaida() {
		verificarConstrucao();
		return numNeuronios;
	}

	@Override
	public boolean temBias() {
		return _bias.isPresent();
	}

	@Override
	public int numParams() {
		verificarConstrucao();

		int parametros = _kernel.tam();
		if (temBias()) parametros += bias().tam();

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
		sb.append(pad).append("Entrada: " + tamEntrada() + "\n");
		sb.append(pad).append("Neurônios: " + numNeuronios() + "\n");
		sb.append(pad).append("Saida: " + tamSaida() + "\n");
		sb.append("\n");

		sb.append(pad + "Pesos:  " + _kernel.shapeStr() + "\n");

		sb.append(pad + "Bias:   ");
		if (temBias()) {
			sb.append(_bias.get().shapeStr() + "\n");
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

		sb.delete(tamanho-1, tamanho);// remover ultimo "\n"    
		
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
		if (temBias()) {
			clone._bias 	= Optional.of(bias());
			clone._gradBias = Optional.of(gradBias());
		}

		clone._entrada = this._entrada.clone();
		clone._kernel = this._kernel.clone();
		clone._buffer = this._buffer.clone();
		clone._saida = this._saida.clone();
		clone._gradSaida = this._gradSaida.clone();
		clone._gradKernel = this._gradKernel.clone();

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
	public int[] shapeEntrada() {
		verificarConstrucao();
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
	public int[] shapeSaida() {
		verificarConstrucao();
		return _saida.shape();
	}

	@Override
	public Tensor kernel() {
		verificarConstrucao();
		return _kernel;
	}

	@Override
	public Variavel[] kernelParaArray() {
		return kernel().paraArray();
	}

	@Override
	public Tensor gradKernel() {
		verificarConstrucao();
		return _gradKernel;
	}

	@Override
	public Variavel[] gradKernelParaArray() {
		return gradKernel().paraArray();
	}

	@Override
	public Tensor bias() {
		verificarConstrucao();

		return _bias.orElseThrow(() -> new IllegalStateException(
			"\nA camada " + nome() + " (" + id + ") não possui bias configurado."
		));
	}

	@Override
	public Variavel[] biasParaArray() {
		return bias().paraArray();
	}

	@Override
	public Tensor gradBias() {
		verificarConstrucao();

		return _gradBias.orElseThrow(() -> new IllegalStateException(
			"\nA camada " + nome() + " (" + id + ") não possui bias configurado."
		));
	}

	@Override
	public Variavel[] gradBiasParaArray() {
		return gradBias().paraArray();
	}

	@Override
	public Tensor gradEntrada() {
		verificarConstrucao();
		return _gradEntrada;
	}

	@Override
	public void setGradKernel(Variavel[] grads) {
		verificarConstrucao();
		_gradKernel.copiarElementos(grads);
	}

	@Override
	public void setGradBias(Variavel[] grads) {
		verificarConstrucao();

		_gradBias.ifPresentOrElse(
			gb -> gb.copiarElementos(grads),
			() -> {
				throw new IllegalStateException(
					"\nA camada " + nome() + " (" + id + ") não possui bias configurado."
				);
			}
		);
	}

	@Override
	public void setKernel(Variavel[] kernel) {
		verificarConstrucao();
		_kernel.copiarElementos(kernel);
	}

	@Override
	public void setBias(Variavel[] bias) {
		verificarConstrucao();

		_bias.ifPresentOrElse(
			b -> b.copiarElementos(bias),
			() -> {
				throw new IllegalStateException(
					"\nA camada " + nome() + " (" + id + ") não possui bias configurado."
				);
			}
		);
	}

}
