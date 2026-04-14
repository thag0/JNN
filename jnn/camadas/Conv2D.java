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
public class Conv2D extends Camada implements Cloneable {

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
	private final int[] shapeFiltro = {1, 1};

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
	public Tensor _entrada;

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
	private boolean usarBias = true;
	
	/**
	 * Tensor contendo os valores de saídas da camada.
	 * <p>
	 *    O formato da saída é dado por:
	 * </p>
	 * <pre>saida = (filtros, altura, largura) </pre>
	 */
	public Tensor _saida;

	/**
	 * Tensor contendo os valores dos gradientes usados para 
	 * a retropropagação para camadas anteriores.
	 * <p>
	 *    O formato do gradiente de entrada é dado por:
	 * </p>
	 * <pre>gradEntrada = (canais, altura, largura) </pre>
	 */
	public Tensor _gradEntrada;

	/**
	 * Tensor contendo os valores dos gradientes relativos a saída
	 * da camada.
	 * <p>
	 *    O formato dos gradientes da saída é dado por:
	 * </p>
	 * <pre>gradSaida = (filtros, altura, largura) </pre>
	 */
	public Tensor _gradSaida;

	/**
	 * Inicializador para os filtros da camada.
	 */
	private Inicializador iniK = new GlorotUniforme();

	/**
	 * Inicializador para o bias da camada.
	 */
	private Inicializador iniB = new Zeros();

	/**
	 * Utilitário.
	 */
	private LayerOps lops = new LayerOps();

	/**
	 * Instancia uma camada Conv2D.
	 * @param entrada formato de entrada (canais, altura, largura).
	 * @param filtros quantidade de filtros.
	 * @param filtro formato dos filtros (altura, largura).
	 * @param pad quantidade de padding aplicado na entrada, pode ser uma {@code String} ("valid" ou "same"),
	 * ou pode ser um valor {@code inteiro} que será aplicado tanto na altura como na largura da entrada.
	 * @param iniK inicializador para os filtros.
	 * @param iniB inicializador para os bias.
	 */
	public Conv2D(int[] entrada, int filtros, int[] filtro, Object pad, Object iniK, Object iniB) {
		this(filtros, filtro, pad, iniK, iniB);
		construir(entrada);
	}

	/**
	 * Instancia uma camada Conv2D.
	 * @param entrada formato de entrada (canais, altura, largura).
	 * @param filtros quantidade de filtros.
	 * @param filtro formato dos filtros (altura, largura).
	 * @param pad quantidade de padding aplicado na entrada, pode ser uma {@code String} ("valid" ou "same"),
	 * ou pode ser um valor inteiro que será aplicado tanto na altura como na largura da entrada.
	 */
	public Conv2D(int[] entrada, int filtros, int[] filtro, Object pad) {
		this(entrada, filtros, filtro, pad, null, null);
	}

	/**
	 * Instancia uma camada Conv2D.
	 * @param entrada formato de entrada (canais, altura, largura).
	 * @param filtros quantidade de filtros.
	 * @param filtro formato dos filtros (altura, largura).
	 */
	public Conv2D(int[] entrada, int filtros, int[] filtro) {
		this(entrada, filtros, filtro, "valid", null, null);
	}

	/**
	 * Instancia uma camada Conv2D.
	 * @param filtros quantidade de filtros.
	 * @param filtro formato dos filtros (altura, largura).
	 * @param pad quantidade de padding aplicado na entrada, pode ser uma {@code String} ("valid" ou "same"),
	 * ou pode ser um valor inteiro que será aplicado tanto na altura como na largura da entrada.
	 * @param iniK inicializador para os filtros.
	 * @param iniB inicializador para os bias.
	 */
	public Conv2D(int filtros, int[] filtro, Object pad, Object iniK, Object iniB) {
		JNNutils.validarNaoNulo(filtro, "filtro == null.");

		//formado dos filtros
		if (filtro.length != 2) {
			throw new IllegalArgumentException(
				"\nO formato dos filtros deve conter 2 elementos (altura, largura), " +
				"recebido: " + filtro.length
			);
		}

		if (!JNNutils.apenasMaiorZero(filtro)) {
			throw new IllegalArgumentException(
				"\nOs valores de formato para os filtros devem ser maiores que zero."
			);      
		}

		if (filtros < 1) {
			throw new IllegalArgumentException(
				"\nO número de filtros deve ser maior que zero, recebido: " + filtros
			);
		}
		
		shapeFiltro[0] = filtro[0];
		shapeFiltro[1] = filtro[1];

		shapeOut[0] = filtros;

		resolverPad(pad);
		
		Dicionario dicio = new Dicionario();
		if (iniK != null) this.iniK = dicio.getInicializador(iniK);
		if (iniB != null) this.iniB = dicio.getInicializador(iniB);
	}

	/**
	 * Instancia uma camada Conv2D.
	 * @param filtros quantidade de filtros.
	 * @param filtro formato dos filtros (altura, largura).
	 * @param pad quantidade de padding aplicado na entrada, pode ser uma {@code String} ("valid" ou "same"),
	 * ou pode ser um valor inteiro que será aplicado tanto na altura como na largura da entrada.
	 * @param iniK inicializador para os filtros.
	 */
	public Conv2D(int filtros, int[] filtro, Object pad, Object iniK) {
		this(filtros, filtro, pad, iniK, null);
	}

	/**
	 * Instancia uma camada Conv2D.
	 * @param filtros quantidade de filtros.
	 * @param filtro formato dos filtros (altura, largura).
	 * @param pad quantidade de padding aplicado na entrada, pode ser uma {@code String} ("valid" ou "same"),
	 * ou pode ser um valor inteiro que será aplicado tanto na altura como na largura da entrada.
	 */
	public Conv2D(int filtros, int[] filtro, Object pad) {
		this(filtros, filtro, pad, null, null);
	}

	/**
	 * Instancia uma camada Conv2D.
	 * @param filtros quantidade de filtros.
	 * @param filtro formato dos filtros (altura, largura).
	 */
	public Conv2D(int filtros, int[] filtro) {
		this(filtros, filtro, null, null, null);
	}

	/**
	 * Calcula os valores de padding.
	 * @param pad pad fornecido.
	 */
	private void resolverPad(Object pad) {
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
					this.shapePad[0] = (shapeFiltro[0] - 1) / 2;
					this.shapePad[1] = (shapeFiltro[1] - 1) / 2;
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
	public void construir(int[] shape) {
		JNNutils.validarNaoNulo(shape, "shape == null.");

		if (shape.length != 3) {
			throw new IllegalArgumentException(
				"\nFormato de entrada para a camada " + nome() + " deve conter três " + 
				"elementos (canais, altura, largura), mas recebido tamanho = " + shape.length
			);
		}

		if (!JNNutils.apenasMaiorZero(shape)) {
			throw new IllegalArgumentException(
				"\nOs valores de dimensões de entrada para a camada " + nome() + 
				" devem ser maiores que zero."
			);
		}

		shapeIn[0] = shape[0];// canais
		shapeIn[1] = shape[1];// altura
		shapeIn[2] = shape[2];// largura

		//dim -> ((entrada + 2*pad - filtro) / stride) + 1
		shapeOut[1] = (shapeIn[1] + 2 * shapePad[0] - shapeFiltro[0]) + 1;
		shapeOut[2] = (shapeIn[2] + 2 * shapePad[1] - shapeFiltro[1]) + 1;

		if (shapeOut[1] < 1 || shapeOut[2] < 1) {
			throw new IllegalArgumentException(
				"\nCamada não pode ser construida:" +
				"\nFormato de entrada " + JNNutils.arrayStr(shape) +
				", filtros " + JNNutils.arrayStr(new int[]{shapeOut[0], shapeFiltro[0], shapeFiltro[1]}) +
				" e padidng " + JNNutils.arrayStr(shapePad) +
				" resultam num formato de saída inválido " + JNNutils.arrayStr(shapeOut)
			);
		}

		int[] shapeKernel = {shapeOut[0], shapeIn[0], shapeFiltro[0], shapeFiltro[1]};
		addParam("kernel", shapeKernel);
		_kernel = _params[0];

		_gradEntrada  = addBuffer("Grad Entrada", shapeIn);// não é passado pro otimizador
		_saida        = addBuffer("Saida", shapeOut);

		if (usarBias) {
			addParam("bias", shapeOut[0]);
			_bias = Optional.of(_params[1]);
		} else {
			_bias = Optional.empty();
		}
		
		_treinavel = true;// camada pode ser treinada.
		construida = true;// camada pode ser usada.
	}

	@Override
	public void initParams() {
		verificarConstrucao();
		
		iniK.forward(_kernel.weight);
		_bias.ifPresent(b -> iniB.forward(b.weight));
	}

	@Override
	public void setBias(boolean usar) {
		this.usarBias = usar;
	}

	@Override
	public void ajustarParaLote(int tamLote) {
		int[] in;
		int[] out;

		if (tamLote == 0) {
			in = shapeIn;
			out = shapeOut;

		} else {
			in = new int[4];
			in[0] = tamLote;
			in[1] = shapeIn[0];
			in[2] = shapeIn[1];
			in[3] = shapeIn[2];

			out = new int[4];
			out[0] = tamLote;
			out[1] = shapeOut[0];
			out[2] = shapeOut[1];
			out[3] = shapeOut[2];
		}
		
		_gradEntrada = addBuffer("Grad Entrada", in);
		_saida 		 = addBuffer("Saida", out);
		
		this.tamLote = tamLote;
	}

	@Override
	public Tensor forward(Tensor x) {
		verificarConstrucao();

		final int numDim = x.numDim();

		if (numDim == 3) {
			validarShapes(x.shape(), shapeIn);
			ajustarParaLote(0);
			
		} else if (numDim == 4) {
			validarShapes(x.shape(), shapeIn);

			int lotes = x.tamDim(0);
			if (lotes != this.tamLote) {
				ajustarParaLote(lotes);
			}
		
		} else {
			throw new UnsupportedOperationException(
				"Esperado tensor " + shapeIn.length + "D (canais, altura, largura) ou " +
				(shapeIn.length + 1) + "D (lotes, canais, altura, largura) " +
				", mas recebido: " + x.numDim() + "D."
			);
		}
		
		_entrada = x.contiguous();

		lops.forwardConv2D(_entrada, _kernel, _bias, _saida, shapePad);

		return _saida;
	}

	@Override
	public Tensor backward(Tensor g) {
		verificarConstrucao();

		_gradSaida = g.contiguous();

		lops.backwardConv2D(
			_entrada,
			_kernel,
			_gradSaida,
			_bias,
			_gradEntrada,
			shapePad
		);
		
		return _gradEntrada;
	}

	@Override
	public void gradZero() {
		verificarConstrucao();

		_kernel.grad.zero();
		_bias.ifPresent(b -> b.grad.zero());
	}

	/**
	 * Retorna a quantidade de filtros presentes na camada.
	 * @return quantiadde de filtros presentes na camada.
	 */
	public int numFiltros() {
		verificarConstrucao();
		return _kernel.weight.tamDim(0);
	}

	@Override
	public Tensor saida() {
		verificarConstrucao();
		return _saida;
	}

	@Override
	public boolean temBias() {
		return _bias.isPresent();
	}

	@Override
	public int numParams() {
		verificarConstrucao();

		int p = 0;
		for (var param : params()) {
			p += param.weight.tam();
		}

		return p;
	}

	@Override
	public float[] saidaParaArray() {
		return saida().array();
	}

	@Override 
	public int tamSaida() {
		return saida().tam();
	}

	@Override
	public String info() {
		verificarConstrucao();

		StringBuilder sb = new StringBuilder();
		String pad = " ".repeat(4);
		
		sb.append(nome()).append(" (id ").append(id).append(") = [\n");

		sb.append(pad).append("Entrada: ").append(JNNutils.arrayStr(shapeIn)).append("\n");
		sb.append(pad).append("Filtros: ").append(numFiltros()).append("\n");
		sb.append(pad).append("Saida: ").append(JNNutils.arrayStr(shapeOut)).append("\n");
		sb.append(pad).append("Padding: ").append(JNNutils.arrayStr(shapePad)).append("\n");
		sb.append("\n");

		sb.append(pad).append(_kernel).append("\n");

		if (temBias()) {
			sb.append(pad).append(_bias.get()).append("\n");
		}

		sb.append("]\n");

		return sb.toString();
	}

	@Override
	public Conv2D clone() {
		verificarConstrucao();

		Conv2D clone = (Conv2D) super.clone();
		clone.usarBias = this.usarBias;
		clone._treinavel = this._treinavel;

		clone._kernel = new Parametro("kernel", _kernel.weight);
		clone._kernel.grad.copiar(_kernel.grad);
		
		clone._gradEntrada = this._gradEntrada.clone();
		
		if (temBias()) {
			clone._bias = Optional.of(new Parametro("bias", _bias.get().weight));
			clone._bias.get().grad.copiar(_bias.get().grad);
		}

		clone._saida = this._saida.clone();

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
		verificarConstrucao();
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
		verificarConstrucao();
		return shapeOut.clone();
	}

	/**
	 * Retorna o formato dos filtros contidos na camada.
	 * @return formato de cada filtro (altura, largura).
	 */
	public int[] shapeKernel() {
		verificarConstrucao();
		return shapeFiltro.clone();
	}

	/**
	 * Retorna o formato do padding usado na camada.
	 * @return formato do padding (altura, largura).
	 */
	public int[] shapePadding() {
		verificarConstrucao();
		return shapePad.clone();
	}

	@Override
	public Tensor bias() {
		verificarConstrucao();

		if (temBias()) return _bias.get().weight;

		throw new RuntimeException(
			"\nA camada " + nome() + " (" + id + ") não possui bias configurado."
		);
	}

	@Override
	public Tensor gradEntrada() {
		verificarConstrucao();
		return _gradEntrada; 
	}

	@Override
	public long tamBytes() {
		long tamVars = super.tamBytes(); //base camada + kernel e bias
		tamVars += 4 * shapeIn.length; 
		tamVars += 4 * shapeOut.length; 
		tamVars += 4 * shapeFiltro.length; 
		tamVars += 2 * shapePad.length; 

		long tamTensores =
		_saida.tamBytes() +
		_gradEntrada.tamBytes();

		return tamVars + tamTensores;
	}

}