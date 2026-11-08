package jnn.layers;

import java.util.Optional;

import jnn.core.Dicionario;
import jnn.core.JNNutils;
import jnn.core.Parametro;
import jnn.core.tensor.Tensor;
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
 *    seus {@code pesos}, adicionando os bias caso sejam configurados.
 * </p>
 */
public class Dense extends Layer implements Cloneable {

	/**
	 * Utilitário.
	 */
	private LayerOps lops = new LayerOps();

	/**
	 * Variável controlador para o tamanho de entrada da camada densa.
	 */
	private int[] shapeIn = { 1 };
	 
	/**
	 * Variável controlador para a quantidade de neurônios (unidades) 
	 * da camada densa.
	 */
	private int[] shapeOut = { 1 };

	/**
	 * Parâmetro contendo os valores dos pesos de cada conexão da
	 * entrada com a saída da camada, com formato:
	 * <pre>kernel = (entrada, neuronios)</pre>
	 */
	public Parametro _kernel;

	/**
	 * Parâmetro contendo o bias da camada, seu formato se dá por:
	 * <pre>bias = (neuronios)</pre>
	 */
	public Optional<Parametro> _bias;

	/**
	 * Auxiliar na verificação do uso do bias na camada.
	 */
	private boolean useBias = true;

	/**
	 * Tensor contendo os valores de entrada da camada, seu formato se dá por:
	 * <pre>
	 *    entrada = (tamEntrada)
	 * </pre>
	 */
	public Tensor _input;

	/**
	 * Tensor contendo os valores de resultado da soma entre os valores 
	 * da matriz de somatório com os valores da matriz de bias da camada, seu 
	 * formato se dá por:
	 * <pre>
	 *    saida = (neuronios)
	 * </pre>
	 */
	public Tensor _output;
	
	/**
	 * Tensor contendo os valores de gradientes de cada neurônio da 
	 * camada, seu formato se dá por:
	 * <pre>
	 *    gradSaida = (neuronios)
	 * </pre>
	 */
	public Tensor _gradOutput;

	/**
	 * Gradientes usados para retropropagar os erros para camadas anteriores.
	 * <p>
	 *    O formato do gradiente de entrada é definido por:
	 * </p>
	 * <pre>
	 *    gradEntrada = (tamEntrada)
	 * </pre>
	 */
	public Tensor _gradInput;

	/**
	 * Inicializador para os pesos da camada.
	 */
	private Inicializador kernelInit = new GlorotUniforme();

	/**
	 * Inicializador para os bias da camada.
	 */
	private Inicializador biasInit = new Zeros();

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param in quantidade de conexões de entrada.
	 * @param out quantidade de neurônios (unidades).
	 * @param initKernel inicializador para os pesos da camada.
	 * @param initBias inicializador para os bias da camada.
	 */
	public Dense(int in, int out, Object initKernel, Object initBias) {
		this(out, initKernel, initBias);
		build(new int[]{ in });// construir automaticamente
	}

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param in quantidade de conexões de entrada.
	 * @param out quantidade de neurônios (unidades).
	 * @param initKernel inicializador para os pesos da camada.
	 */
	public Dense(int in, int out, Object initKernel) {
		this(in, out, initKernel, null);
	}

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param in quantidade de conexões de entrada.
	 * @param out quantidade de neurônios (unidades).
	 */
	public Dense(int in, int out) {
		this(in, out, null, null);
	}

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param out quantidade de neurônios (unidades).
	 * @param initKernel inicializador para os pesos da camada.
	 * @param initBias inicializador para os bias da camada.
	 */
	public Dense(int out, Object initKernel, Object initBias) {
		if (out < 1) {
			throw new IllegalArgumentException(
				"\nA camada deve conter ao menos um neurônio."
			);
		}

		shapeOut[0] = out;

		//usar os valores padrão se necessário
		Dicionario dic = new Dicionario();
		if (initKernel != null) this.kernelInit = dic.getInicializador(initKernel);
		if (initBias != null)  this.biasInit = dic.getInicializador(initBias);
	}

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param out quantidade de neurônios (unidades).
	 * @param initKernel inicializador para os pesos da camada.
	 */
	public Dense(int out, Object initKernel) {
		this(out, initKernel, null);
	}

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param out quantidade de neurônios (unidades).
	 */
	public Dense(int out) {
		this(out, null, null);
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
	 *camada.build(shape);
	 * </pre>
	 */
	@Override
	public void build(int[] shape) {
		JNNutils.validarNaoNulo(shape, "shape == null.");

		if (shape.length != 1) {
			throw new IllegalArgumentException(
				"\nFormato de entrada deve conter 1 elemento, recebido " + shape.length
			);
		}

		if (shape[0] < 1) {
			throw new IllegalArgumentException(
				"\nTamanho de entrada deve ser maior que zero."
			);
		}

		shapeIn[0] = shape[0];

		if (shapeOut[0] < 1) {
			throw new IllegalArgumentException(
				"\nNúmero de neurônios para a camada Densa não foi definido."
			);
		}
		
		_kernel = addParam("kernel", shapeIn[0], shapeOut[0]);
		
		if (useBias) _bias = Optional.of(addParam("bias", shapeOut[0]));
		else _bias = Optional.empty();

		_gradInput = addBuffer("grad input", shapeIn[0]);
		_output    = addBuffer("output", shapeOut[0]);
		
		_treinable = true;// camada pode ser treinada.
		_built = true;// camada pode ser usada.
	}

	@Override
	public void initParams() {
		isBuilt();

		kernelInit.forward(_kernel.weight);
		_bias.ifPresent(b -> biasInit.forward(b.weight));
	}

	@Override
	public void setBias(boolean useBias) {
		this.useBias = useBias;
	}

	@Override
	public void fitToBatch(int batchSize) {
		int[] shapeIn, out;

		if (batchSize == 0) {
			shapeIn = new int[]{ sizeInput() };
			out = shapeOut;
			
		} else {
			shapeIn = new int[]{ batchSize, sizeInput() };
			out = new int[]{ batchSize, numNeurons() };
		}

		_gradInput = addBuffer("grad input", shapeIn);
		_output    = addBuffer("output", out);
		
		this.batchSize = batchSize;
	}

	@Override
	public Tensor forward(Tensor x) {
		isBuilt();

		if (x.numDim() == 1) {
			validateShapes(x.shape(), shapeIn);
			fitToBatch(0);
			
		} else if (x.numDim() == 2) {
			validateShapes(x.shape(), shapeIn);
			int batches = x.lenDim(0);

			if (this.batchSize != batches) fitToBatch(batches);
			
		} else {
			throw new UnsupportedOperationException(
				"\n Dimensões de X " + x.numDim() + " não suportadas."
			);
		}

		_input = x.contiguous();

		lops.forwardDense(_input, _kernel, _bias, _output);

		return _output;
	}

	@Override
	public Tensor backward(Tensor g) {
		isBuilt();
		
		_gradOutput = g.contiguous();

		lops.backwardDense(
			_input,
			_kernel,
			_gradOutput,
			_bias,
			_gradInput
		);

		return _gradInput;
	}

	@Override
	public void zeroGrad() {
		isBuilt();

		_kernel.grad.zero();
		_bias.ifPresent(b -> b.grad.zero());
	}

	@Override
	public Tensor output() {
		isBuilt();
		return _output;
	}

	/**
	 * Retorna a quantidade de neurônios presentes na camada.
	 * @return quantidade de neurônios presentes na camada.
	 */
	public int numNeurons() {
		isBuilt();
		return shapeOut[0];
	}

	/**
	 * Retorna a capacidade de entrada da camada.
	 * @return tamanho de entrada da camada.
	 */
	public int sizeInput() {
		isBuilt();
		return shapeIn[0];
	}

	/**
	 * Retorna a capacidade de saída da camada.
	 * @return tamanho de saída da camada.
	 */
	public int sizeOutput() {
		isBuilt();
		return numNeurons();
	}

	@Override
	public boolean hasBias() {
		return _bias.isPresent();
	}

	@Override
	public int numParams() {
		isBuilt();

		int p = 0;
		for (var param : params()) {
			p += param.weight.len();
		}

		return p;
	}

	@Override
	public String info() {
		isBuilt();

		StringBuilder sb = new StringBuilder();
		String pad = " ".repeat(4);
		
		sb.append(name()).append("(\n");

		sb.append(pad).append("In: ").append(JNNutils.arrayStr(shapeIn)).append("\n");
		sb.append(pad).append("Out: ").append(JNNutils.arrayStr(shapeOut)).append("\n");

		sb.append(pad).append(_kernel).append("\n");
		if (hasBias()) sb.append(pad).append(_bias.get()).append("\n");

		sb.append(")");

		return sb.toString();
	}

	@Override
	public Dense clone() {
		isBuilt();

		Dense clone = (Dense) super.clone();

		clone.lops = new LayerOps();
		clone._treinable = this._treinable;

		clone._kernel = _kernel.clone();

		clone.useBias = this.useBias;
		if (hasBias()) {
			clone._bias = Optional.of(_bias.get().clone());
		}

		clone._output = this._output.clone();

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
	public int[] shapeIn() {
		isBuilt();
		return _gradInput.shape();
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
	public int[] shapeOut() {
		isBuilt();
		return _output.shape();
	}

	@Override
	public Tensor bias() {
		isBuilt();

		if (hasBias()) return _bias.get().weight;

		throw new RuntimeException(
			"\nA camada " + name() + " (" + id + ") não possui bias configurado."
		);
	}

	@Override
	public Tensor gradInput() {
		isBuilt();
		return _gradInput;
	}

	@Override
	public long sizeBytes() {
		long tamVars = super.sizeBytes(); //base camada + tensores
		tamVars += 4; //tamEntrada
		tamVars += 4; //numNeuronios
		tamVars += 1; //usarBias

		long tamTensores =
		_output.sizeBytes() +
		_gradInput.sizeBytes();

		return tamVars + tamTensores;
	}

}
