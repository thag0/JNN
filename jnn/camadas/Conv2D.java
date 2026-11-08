package jnn.camadas;

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
 *		Camada Convolucional
 * </h2>
 * <p>
 *		Aplica uma convolução bidimensional sobre um tensor de entrada composto
 *		por múltiplos canais espaciais. A camada utiliza um conjunto de filtros
 *		(kernels) deslizantes que percorrem as dimensões espaciais da entrada
 *		(altura e largura), produzindo um conjunto de mapas de características
 *		como saída.
 * </p>
 * <p>
 *		Caso o bias seja configurado, ele será adicionado na saída da camada.
 * </p>
 * @see <a href="https://github.com/thag0/JNN/blob/main/jnn/camadas/Conv2D.java"> Conv2D </a>
 */
public class Conv2D extends Layer implements Cloneable {

	/**
	 * Formato de entrada da camada convolucional, dado por:
	 * <pre>
	 *    form = (canais, altura, largura)
	 * </pre>
	 */
	private final int[] shapeIn = {1, 1, 1};

	/**
	 * Formato de cada filtro da camada convolucional, dado por:
	 * <pre>
	 *    form = (altura, largura)
	 * </pre>
	 */
	private final int[] shapeKernel = {1, 1};

	/**
	 * Formato de saída da camada convolucional, dado por:
	 * <pre>
	 *    form = (filtros, altura, largura)
	 * </pre>
	 */
	private final int[] shapeOut = {1, 1, 1};

	/**
	 * Formato de padding aplicado na entrada da camada antes de calcular a saída.
	 * <pre>
	 *    pad = (altura, largura)
	 * </pre>
	 */
	private final int[] shapePad = {1, 1};

	/**
	 * Tensor contendo os valores de entrada para a camada,
	 * que serão usados para o processo de feedforward.
	 * <p>
	 *    O formato da entrada é dado por:
	 * </p>
	 * <pre>entrada = (canais, altura, largura) </pre>
	 */
	public Tensor _input;

	/**
	 * Tensor contendo o kernel (ou filtros) da camada.
	 * <p>
	 *    O formato dos filtros é dado por:
	 * </p>
	 * <pre>kernel = (filtros, canais, altura, largura) </pre>
	 */
	public Parametro _kernel;

	/**
	 * Tensor contendo os bias (vieses) para cada valor de 
	 * saída da camada.
	 * <p>
	 *    O formato do bias é dado por:
	 * </p>
	 * <pre>bias = (filtros) </pre>
	 */
	public Optional<Parametro> _bias;

	/**
	 * Auxiliar na verificação de uso do bias.
	 */
	private boolean useBias = true;
	
	/**
	 * Tensor contendo os valores de saídas da camada.
	 * <p>
	 *    O formato da saída é dado por:
	 * </p>
	 * <pre>saida = (filtros, altura, largura) </pre>
	 */
	public Tensor _output;

	/**
	 * Tensor contendo os valores dos gradientes usados para 
	 * a retropropagação para camadas anteriores.
	 * <p>
	 *    O formato do gradiente de entrada é dado por:
	 * </p>
	 * <pre>gradEntrada = (canais, altura, largura) </pre>
	 */
	public Tensor _gradInput;

	/**
	 * Tensor contendo os valores dos gradientes relativos a saída
	 * da camada.
	 * <p>
	 *    O formato dos gradientes da saída é dado por:
	 * </p>
	 * <pre>gradSaida = (filtros, altura, largura) </pre>
	 */
	public Tensor _gradOutput;

	/**
	 * Inicializador para os filtros da camada.
	 */
	private Inicializador kernelInit = new GlorotUniforme();

	/**
	 * Inicializador para o bias da camada.
	 */
	private Inicializador biasInit = new Zeros();

	/**
	 * Utilitário.
	 */
	private LayerOps lops = new LayerOps();

	/**
	 * Instancia uma camada Conv2D.
	 * @param input formato de entrada (canais, altura, largura).
	 * @param filters quantidade de filtros.
	 * @param filterSize formato dos filtros (altura, largura).
	 * @param pad quantidade de padding aplicado na entrada, pode ser uma {@code String} ("valid" ou "same"),
	 * ou pode ser um valor {@code inteiro} que será aplicado tanto na altura como na largura da entrada.
	 * @param initKernel inicializador para os filtros.
	 * @param initBias inicializador para os bias.
	 */
	public Conv2D(int[] input, int filters, int[] filterSize, Object pad, Object initKernel, Object initBias) {
		this(filters, filterSize, pad, initKernel, initBias);
		build(input);
	}

	/**
	 * Instancia uma camada Conv2D.
	 * @param input formato de entrada (canais, altura, largura).
	 * @param filters quantidade de filtros.
	 * @param filterSize formato dos filtros (altura, largura).
	 * @param pad quantidade de padding aplicado na entrada, pode ser uma {@code String} ("valid" ou "same"),
	 * ou pode ser um valor inteiro que será aplicado tanto na altura como na largura da entrada.
	 */
	public Conv2D(int[] input, int filters, int[] filterSize, Object pad) {
		this(input, filters, filterSize, pad, null, null);
	}

	/**
	 * Instancia uma camada Conv2D.
	 * @param input formato de entrada (canais, altura, largura).
	 * @param filters quantidade de filtros.
	 * @param filterSize formato dos filtros (altura, largura).
	 */
	public Conv2D(int[] input, int filters, int[] filterSize) {
		this(input, filters, filterSize, "valid", null, null);
	}

	/**
	 * Instancia uma camada Conv2D.
	 * @param filters quantidade de filtros.
	 * @param filterSize formato dos filtros (altura, largura).
	 * @param pad quantidade de padding aplicado na entrada, pode ser uma {@code String} ("valid" ou "same"),
	 * ou pode ser um valor inteiro que será aplicado tanto na altura como na largura da entrada.
	 * @param initKernel inicializador para os filtros.
	 * @param initBias inicializador para os bias.
	 */
	public Conv2D(int filters, int[] filterSize, Object pad, Object initKernel, Object initBias) {
		JNNutils.validarNaoNulo(filterSize, "filtro == null.");

		//formado dos filtros
		if (filterSize.length != 2) {
			throw new IllegalArgumentException(
				"\nO formato dos filtros deve conter 2 elementos (altura, largura), " +
				"recebido: " + filterSize.length
			);
		}

		if (!JNNutils.apenasMaiorZero(filterSize)) {
			throw new IllegalArgumentException(
				"\nOs valores de formato para os filtros devem ser maiores que zero."
			);      
		}

		if (filters < 1) {
			throw new IllegalArgumentException(
				"\nO número de filtros deve ser maior que zero, recebido: " + filters
			);
		}
		
		shapeKernel[0] = filterSize[0];
		shapeKernel[1] = filterSize[1];

		shapeOut[0] = filters;

		resolvePad(pad);
		
		Dicionario dicio = new Dicionario();
		if (initKernel != null) this.kernelInit = dicio.getInicializador(initKernel);
		if (initBias != null) this.biasInit = dicio.getInicializador(initBias);
	}

	/**
	 * Instancia uma camada Conv2D.
	 * @param filters quantidade de filtros.
	 * @param filterSize formato dos filtros (altura, largura).
	 * @param pad quantidade de padding aplicado na entrada, pode ser uma {@code String} ("valid" ou "same"),
	 * ou pode ser um valor inteiro que será aplicado tanto na altura como na largura da entrada.
	 * @param initKernel inicializador para os filtros.
	 */
	public Conv2D(int filters, int[] filterSize, Object pad, Object initKernel) {
		this(filters, filterSize, pad, initKernel, null);
	}

	/**
	 * Instancia uma camada Conv2D.
	 * @param filters quantidade de filtros.
	 * @param filterSize formato dos filtros (altura, largura).
	 * @param pad quantidade de padding aplicado na entrada, pode ser uma {@code String} ("valid" ou "same"),
	 * ou pode ser um valor inteiro que será aplicado tanto na altura como na largura da entrada.
	 */
	public Conv2D(int filters, int[] filterSize, Object pad) {
		this(filters, filterSize, pad, null, null);
	}

	/**
	 * Instancia uma camada Conv2D.
	 * @param filters quantidade de filtros.
	 * @param filterSize formato dos filtros (altura, largura).
	 */
	public Conv2D(int filters, int[] filterSize) {
		this(filters, filterSize, null, null, null);
	}

	/**
	 * Calcula os valores de padding.
	 * @param pad pad fornecido.
	 */
	private void resolvePad(Object pad) {
		if (pad == null) { //valid de padrão
			this.shapePad[0] = 0;
			this.shapePad[1] = 0;
		
		} else if (pad instanceof String p) {
			p = p.toLowerCase();
			switch (p) {
				case "valid" -> {
					this.shapePad[0] = 0;
					this.shapePad[1] = 0;
				}

				case "same" -> {
					this.shapePad[0] = (shapeKernel[0] - 1) / 2;
					this.shapePad[1] = (shapeKernel[1] - 1) / 2;
				}

				default -> throw new IllegalArgumentException(
					"\nTipo de padding inválido: \"" + shapePad + "\"." +
					"\nUse \"valid\", \"same\" ou valores inteiros."
				);
			}
		
		} else if (pad instanceof Integer p) {
			if (p < 0) {
				throw new IllegalArgumentException(
					"\nValores de padding devem ser maiores que 0, recebido " + p
				);
			}

			this.shapePad[0] = p;
			this.shapePad[1] = p;
		
		} else {
			throw new IllegalArgumentException(
				"\nTipo de objeto para padding " + pad.getClass() + " inválido."
			);
		}
	}

	@Override
	public void build(int[] shape) {
		JNNutils.validarNaoNulo(shape, "shape == null.");

		if (shape.length != 3) {
			throw new IllegalArgumentException(
				"\nFormato de entrada para a camada " + name() + " deve conter três " + 
				"elementos (canais, altura, largura), mas recebido tamanho = " + shape.length
			);
		}

		if (!JNNutils.apenasMaiorZero(shape)) {
			throw new IllegalArgumentException(
				"\nOs valores de dimensões de entrada para a camada " + name() + 
				" devem ser maiores que zero."
			);
		}

		shapeIn[0] = shape[0];// canais
		shapeIn[1] = shape[1];// altura
		shapeIn[2] = shape[2];// largura

		//dim -> ((entrada + 2*pad - filtro) / stride) + 1
		shapeOut[1] = (shapeIn[1] + 2 * shapePad[0] - shapeKernel[0]) + 1;
		shapeOut[2] = (shapeIn[2] + 2 * shapePad[1] - shapeKernel[1]) + 1;

		if (shapeOut[1] < 1 || shapeOut[2] < 1) {
			throw new IllegalArgumentException(
				"\nCamada não pode ser construida:" +
				"\nFormato de entrada " + JNNutils.arrayStr(shape) +
				", filtros " + JNNutils.arrayStr(new int[]{shapeOut[0], shapeKernel[0], shapeKernel[1]}) +
				" e padidng " + JNNutils.arrayStr(shapePad) +
				" resultam num formato de saída inválido " + JNNutils.arrayStr(shapeOut)
			);
		}

		_kernel = addParam(
			"kernel", 
			shapeOut[0], shapeIn[0], shapeKernel[0], shapeKernel[1]
		);
		
		if (useBias) _bias = Optional.of(addParam("bias", shapeOut[0]));
		else _bias = Optional.empty();

		_gradInput = addBuffer("grad input", shapeIn);// não é passado pro otimizador
		_output    = addBuffer("output", shapeOut);
		
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
		int[] in;
		int[] out;

		if (batchSize == 0) {
			in = shapeIn;
			out = shapeOut;

		} else {
			in = new int[] {
				batchSize,
				shapeIn[0],
				shapeIn[1],
				shapeIn[2]
			};

			out = new int[] {
				batchSize,
				shapeOut[0],
				shapeOut[1],
				shapeOut[2]
			};
		}
		
		_gradInput = addBuffer("grad input", in);
		_output    = addBuffer("output", out);
		
		this.batchSize = batchSize;
	}

	@Override
	public Tensor forward(Tensor x) {
		isBuilt();

		final int numDim = x.numDim();

		if (numDim == 3) {
			validateShapes(x.shape(), shapeIn);
			fitToBatch(0);
			
		} else if (numDim == 4) {
			validateShapes(x.shape(), shapeIn);

			int lotes = x.lenDim(0);
			if (lotes != this.batchSize) {
				fitToBatch(lotes);
			}
		
		} else {
			throw new UnsupportedOperationException(
				"Esperado tensor " + shapeIn.length + "D (canais, altura, largura) ou " +
				(shapeIn.length + 1) + "D (lotes, canais, altura, largura) " +
				", mas recebido: " + x.numDim() + "D."
			);
		}
		
		_input = x.contiguous();

		lops.forwardConv2D(_input, _kernel, _bias, _output, shapePad);

		return _output;
	}

	@Override
	public Tensor backward(Tensor g) {
		isBuilt();

		_gradOutput = g.contiguous();

		lops.backwardConv2D(
			_input,
			_kernel,
			_gradOutput,
			_bias,
			_gradInput,
			shapePad
		);
		
		return _gradInput;
	}

	@Override
	public void zeroGrad() {
		isBuilt();

		_kernel.grad.zero();
		_bias.ifPresent(b -> b.grad.zero());
	}

	/**
	 * Retorna a quantidade de filtros presentes na camada.
	 * @return quantiadde de filtros presentes na camada.
	 */
	public int numFilters() {
		isBuilt();
		return _kernel.weight.lenDim(0);
	}

	@Override
	public Tensor output() {
		isBuilt();
		return _output;
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
	public float[] outputToArray() {
		return output().array();
	}

	@Override 
	public int sizeOutput() {
		return output().len();
	}

	@Override
	public String info() {
		isBuilt();

		StringBuilder sb = new StringBuilder();
		String pad = " ".repeat(4);
		
		sb.append(name()).append("(").append("\n");

		sb.append(pad).append("In: ").append(JNNutils.arrayStr(shapeIn)).append("\n");
		sb.append(pad).append("Out: ").append(JNNutils.arrayStr(shapeOut)).append("\n");
		sb.append(pad).append("Padding: ").append(JNNutils.arrayStr(shapePad)).append("\n");

		sb.append(pad).append(_kernel).append("\n");
		if (hasBias()) sb.append(pad).append(_bias.get()).append("\n");

		sb.append(")");

		return sb.toString();
	}

	@Override
	public Conv2D clone() {
		isBuilt();

		Conv2D clone = (Conv2D) super.clone();
		clone.useBias = this.useBias;
		clone._treinable = this._treinable;

		clone._kernel = _kernel.clone();
		
		clone._gradInput = this._gradInput.clone();
		
		if (hasBias()) {
			clone._bias = Optional.of(_bias.get().clone());
		}

		clone._output = this._output.clone();

		return clone;
	}

	/**
	 * Calcula o formato de entrada da camada Convolucional, que é disposto da
	 * seguinte forma:
	 * <pre>
	 *    formato = (canais, altura, largura)
	 * </pre>
	 * @return formato de entrada da camada.
	 */
	@Override
	public int[] shapeIn() {
		isBuilt();
		return shapeIn.clone();
	}
 
	/**
	 * Calcula o formato de saída da camada Convolucional, que é disposto da
	 * seguinte forma:
	 * <pre>
	 *    formato = (canais, altura, largura)
	 * </pre>
	 * @return formato de saída da camada.
	 */
	@Override
	public int[] shapeOut() {
		isBuilt();
		return shapeOut.clone();
	}

	/**
	 * Retorna o formato dos filtros contidos na camada.
	 * @return formato de cada filtro (altura, largura).
	 */
	public int[] shapeKernel() {
		isBuilt();
		return shapeKernel.clone();
	}

	/**
	 * Retorna o formato do padding usado na camada.
	 * @return formato do padding (altura, largura).
	 */
	public int[] shapePadding() {
		isBuilt();
		return shapePad.clone();
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
		long tamVars = super.sizeBytes(); //base camada + kernel e bias
		tamVars += 4 * shapeIn.length; 
		tamVars += 4 * shapeOut.length; 
		tamVars += 4 * shapeKernel.length; 
		tamVars += 2 * shapePad.length; 

		long tamTensores =
		_output.sizeBytes() +
		_gradInput.sizeBytes();

		return tamVars + tamTensores;
	}

}