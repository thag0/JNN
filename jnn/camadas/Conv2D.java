package jnn.camadas;

import java.util.Optional;

import jnn.acts.Ativacao;
import jnn.acts.Linear;
import jnn.core.Dicionario;
import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;
import jnn.inicializadores.GlorotUniforme;
import jnn.inicializadores.Inicializador;
import jnn.inicializadores.Zeros;

/**
 * <h2>
 *    Camada Convolucional
 * </h2>
 * <p>
 *    A camada convolucional realiza operações de convolução sobre a entrada
 *    utilizando filtros (kernels) para extrair características locais, dada 
 *    pela expressão:
 * </p>
 * <pre>
 *buffer = conv2D(entrada, filtros);
 *buffer.add(bias);
 * </pre>
 * Após a propagação dos dados, a função de ativação da camada é aplicada ao 
 * resultado do somatório, que por fim é salvo na saída da camada.
 * <pre>
 *    saida = ativacao(buffer);
 * </pre>
 * <h3>
 *    Detalhe adicional:
 * </h3>
 * Na realidade a operação realizada dentro da camada convolucional é chamada de
 * correlação cruzada, é nela que aplicamos os kernels pela entrada recebida. A 
 * operação de convolução tem a peculiaridade de rotacionar o filtro 180° antes 
 * de ser executada.
 */
public class Conv2D extends Camada implements Cloneable {

	/**
	 * Utilitário.
	 */
	private LayerOps lops = new LayerOps();

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
	 * <pre>kernel = (filtros, canais, altFiltro, largFiltro) </pre>
	 */
	public Tensor _kernel;

	/**
	 * Tensor contendo os bias (vieses) para cada valor de 
	 * saída da camada.
	 * <p>
	 *    O formato do bias é dado por:
	 * </p>
	 * <pre>bias = (filtros) </pre>
	 */
	public Optional<Tensor> _bias;

	/**
	 * Auxiliar na verificação de uso do bias.
	 */
	private boolean usarBias = true;

	/**
	 * Auxilar no controle de treinamento em lotes.
	 */
	private int tamLote;

	/**
	 * Tensor contendo os valores de resultados intermediários 
	 * do processamento da camada.
	 * <p>
	 *    O formato buffer é dado por:
	 * </p>
	 * <pre>buffer = (filtros, altSaida, largSaida) </pre>
	 */
	public Tensor _buffer;
	
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
	 * Tensor contendo os valores dos gradientes relativos a cada
	 * filtro da camada.
	 * <p>
	 *    O formato dos gradientes para os filtros é dado por:
	 * </p>
	 * <pre>gradFiltros = (filtros, canais, altFiltro, largFiltro) </pre>
	 */
	public Tensor _gradKernel;

	/**
	 * Tensor contendo os valores dos gradientes relativos a cada
	 * bias da camada.
	 * <p>
	 *    O formato dos gradientes para os bias é dado por:
	 * </p>
	 * <pre>gradBias = (numFiltros) </pre>
	 */
	public Optional<Tensor> _gradBias;

	/**
	 * Função de ativação da camada.
	 */
	private Ativacao act = new Linear();

	/**
	 * Inicializador para os filtros da camada.
	 */
	private Inicializador iniKernel = new GlorotUniforme();

	/**
	 * Inicializador para os bias da camada.
	 */
	private Inicializador iniBias = new Zeros();

	/**
	 * Instancia uma camada convolucional de acordo com os formatos fornecidos.
	 * <p>
	 *    A disposição do formato de entrada deve ser da seguinte forma:
	 * </p>
	 * <pre>
	 *    formEntrada = (canais, altura, largura)
	 * </pre>
	 * Onde largura e altura devem corresponder as dimensões dos dados de entrada
	 * que serão processados pela camada e a profundidade diz respeito a quantidade
	 * de entradas que a camada deve processar.
	 * <p>
	 *    A disposição do formato do filtro deve ser da seguinte forma:
	 * </p>
	 * <pre>
	 *    formFiltro = (altura, largura)
	 * </pre>
	 * Onde largura e altura correspondem as dimensões que os filtros devem assumir.
	 * @param entrada formato de entrada da camada.
	 * @param filtros quantidade de filtros.
	 * @param filtro formato dos filtros da camada (altura, largura).
	 * @param act função de ativação.
	 * @param iniKernel inicializador para os filtros.
	 * @param iniBias inicializador para os bias.
	 */
	public Conv2D(int[] entrada, int filtros, int[] filtro, Object act, Object iniKernel, Object iniBias) {
		this(filtros, filtro, act, iniKernel, iniBias);
		construir(entrada);
	}

	/**
	 * Instancia uma camada convolucional de acordo com os formatos fornecidos.
	 * <p>
	 *    A disposição do formato de entrada deve ser da seguinte forma:
	 * </p>
	 * <pre>
	 *    formEntrada = (canais, altura, largura)
	 * </pre>
	 * Onde largura e altura devem corresponder as dimensões dos dados de entrada
	 * que serão processados pela camada e a profundidade diz respeito a quantidade
	 * de entradas que a camada deve processar.
	 * <p>
	 *    A disposição do formato do filtro deve ser da seguinte forma:
	 * </p>
	 * <pre>
	 *    formFiltro = (altura, largura)
	 * </pre>
	 * Onde largura e altura correspondem as dimensões que os filtros devem assumir.
	 * @param entrada formato de entrada da camada.
	 * @param filtros quantidade de filtros.
	 * @param filtro formato dos filtros da camada (altura, largura).
	 * @param act função de ativação.
	 * @param iniKernel inicializador para os filtros.
	 */
	public Conv2D(int[] entrada, int filtros, int[] filtro, Object act, Object iniKernel) {
		this(entrada, filtros, filtro, act, iniKernel, null);
	}

	/**
	 * Instancia uma camada convolucional de acordo com os formatos fornecidos.
	 * <p>
	 *    A disposição do formato de entrada deve ser da seguinte forma:
	 * </p>
	 * <pre>
	 *    formEntrada = (canais, altura, largura)
	 * </pre>
	 * Onde largura e altura devem corresponder as dimensões dos dados de entrada
	 * que serão processados pela camada e a profundidade diz respeito a quantidade
	 * de entradas que a camada deve processar.
	 * <p>
	 *    A disposição do formato do filtro deve ser da seguinte forma:
	 * </p>
	 * <pre>
	 *    formFiltro = (altura, largura)
	 * </pre>
	 * Onde largura e altura correspondem as dimensões que os filtros devem assumir.
	 * @param entrada formato de entrada da camada.
	 * @param filtros quantidade de filtros.
	 * @param filtro formato dos filtros da camada (altura, largura).
	 * @param act função de ativação.
	 */
	public Conv2D(int[] entrada, int filtros, int[] filtro, Object act) {
		this(entrada, filtros, filtro, act, null, null);
	}

	/**
	 * Instancia uma camada convolucional de acordo com os formatos fornecidos.
	 * <p>
	 *    A disposição do formato de entrada deve ser da seguinte forma:
	 * </p>
	 * <pre>
	 *    formEntrada = (canais, altura, largura)
	 * </pre>
	 * Onde largura e altura devem corresponder as dimensões dos dados de entrada
	 * que serão processados pela camada e a profundidade diz respeito a quantidade
	 * de entradas que a camada deve processar.
	 * <p>
	 *    A disposição do formato do filtro deve ser da seguinte forma:
	 * </p>
	 * <pre>
	 *    formFiltro = (altura, largura)
	 * </pre>
	 * Onde largura e altura correspondem as dimensões que os filtros devem assumir.
	 * <p>
	 *    O valor de uso do bias será usado como {@code true} por padrão.
	 * <p>
	 * @param entrada formato de entrada da camada.
	 * @param filtros quantidade de filtros.
	 * @param filtro formato dos filtros da camada (altura, largura).
	 */
	public Conv2D(int[] entrada, int filtros, int[] filtro) {
		this(entrada, filtros, filtro, null, null, null);
	}

	/**
	 * Instancia uma camada convolucional de acordo com os formatos fornecidos.
	 * <p>
	 *    A disposição do formato de entrada deve ser da seguinte forma:
	 * </p>
	 * <pre>
	 *    formEntrada = (canais, altura, largura)
	 * </pre>
	 * Onde largura e altura devem corresponder as dimensões dos dados de entrada
	 * que serão processados pela camada e a profundidade diz respeito a quantidade
	 * de entradas que a camada deve processar.
	 * <p>
	 *    A disposição do formato do filtro deve ser da seguinte forma:
	 * </p>
	 * <pre>
	 *    formFiltro = (altura, largura)
	 * </pre>
	 * Onde largura e altura correspondem as dimensões que os filtros devem assumir.
	 * @param filtro formato dos filtros da camada.
	 * @param filtros quantidade de filtros.
	 * @param act função de ativação.
	 * @param iniKernel inicializador para os filtros.
	 * @param iniBias inicializador para os bias.
	 */
	public Conv2D(int filtros, int[] filtro, Object act, Object iniKernel, Object iniBias) {
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
		
		Dicionario dicio = new Dicionario();
		if (act != null) this.act = dicio.getAtivacao(act);
		if (iniKernel != null) this.iniKernel = dicio.getInicializador(iniKernel);
		if (iniBias != null) this.iniBias = dicio.getInicializador(iniBias);
	}

	/**
	 * Instancia uma camada convolucional de acordo com os formatos fornecidos.
	 * <p>
	 *    A disposição do formato de entrada deve ser da seguinte forma:
	 * </p>
	 * <pre>
	 *    formEntrada = (canais, altura, largura)
	 * </pre>
	 * Onde largura e altura devem corresponder as dimensões dos dados de entrada
	 * que serão processados pela camada e a profundidade diz respeito a quantidade
	 * de entradas que a camada deve processar.
	 * <p>
	 *    A disposição do formato do filtro deve ser da seguinte forma:
	 * </p>
	 * <pre>
	 *    formFiltro = (altura, largura)
	 * </pre>
	 * Onde largura e altura correspondem as dimensões que os filtros devem assumir.
	 * @param filtro formato dos filtros da camada.
	 * @param filtros quantidade de filtros.
	 * @param act função de ativação.
	 * @param iniKernel inicializador para os filtros.
	 */
	public Conv2D(int filtros, int[] filtro, Object act, Object iniKernel) {
		this(filtros, filtro, act, iniKernel, null);
	}

	/**
	 * Instancia uma camada convolucional de acordo com os formatos fornecidos.
	 * <p>
	 *    A disposição do formato de entrada deve ser da seguinte forma:
	 * </p>
	 * <pre>
	 *    formEntrada = (canais, altura, largura)
	 * </pre>
	 * Onde largura e altura devem corresponder as dimensões dos dados de entrada
	 * que serão processados pela camada e a profundidade diz respeito a quantidade
	 * de entradas que a camada deve processar.
	 * <p>
	 *    A disposição do formato do filtro deve ser da seguinte forma:
	 * </p>
	 * <pre>
	 *    formFiltro = (altura, largura)
	 * </pre>
	 * Onde largura e altura correspondem as dimensões que os filtros devem assumir.
	 * @param filtro formato dos filtros da camada.
	 * @param filtros quantidade de filtros.
	 * @param act função de ativação.
	 */
	public Conv2D(int filtros, int[] filtro, Object act) {
		this(filtros, filtro, act, null, null);
	}

	/**
	 * Instancia uma camada convolucional de acordo com os formatos fornecidos.
	 * <p>
	 *    A disposição do formato de entrada deve ser da seguinte forma:
	 * </p>
	 * <pre>
	 *    formEntrada = (canais, altura, largura)
	 * </pre>
	 * Onde largura e altura devem corresponder as dimensões dos dados de entrada
	 * que serão processados pela camada e a profundidade diz respeito a quantidade
	 * de entradas que a camada deve processar.
	 * <p>
	 *    A disposição do formato do filtro deve ser da seguinte forma:
	 * </p>
	 * <pre>
	 *    formFiltro = (altura, largura)
	 * </pre>
	 * Onde largura e altura correspondem as dimensões que os filtros devem assumir.
	 * @param filtro formato dos filtros da camada.
	 * @param filtros quantidade de filtros.
	 */
	public Conv2D(int filtros, int[] filtro) {
		this(filtros, filtro, null, null, null);
	}
	
	/**
	 * Inicializa os parâmetros necessários para a camada Convolucional.
	 * <p>
	 *    O formato de entrada deve ser um array contendo o tamanho de 
	 *    cada dimensão de entrada da camada, e deve estar no formato:
	 * </p>
	 * <pre>
	 *    entrada = (canais, altura, largura)
	 * </pre>
	 * @param shape formato de entrada para a camada.
	 */
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

		//dim -> ((entrada - filtro) / stride) + 1
		shapeOut[1] = shapeIn[1] - shapeFiltro[0] + 1;
		shapeOut[2] = shapeIn[2] - shapeFiltro[1] + 1;

		if (shapeOut[1] < 1 || shapeOut[2] < 1) {
			throw new IllegalArgumentException(
				"\nCamada não pode ser construida:" +
				"\nFormato de entrada " + JNNutils.arrayStr(shape) +
				" e formato dos filtros " + 
				JNNutils.arrayStr(new int[]{shapeOut[0], shapeFiltro[0], shapeFiltro[1]}) +
				" resultam num formato de saída inválido " + JNNutils.arrayStr(shapeOut)
			);
		}

		_gradEntrada  = addParam("Grad Entrada", shapeIn);
		_kernel       = addParam("Kernel", shapeOut[0], shapeIn[0], shapeFiltro[0], shapeFiltro[1]);
		_gradKernel   = addParam("Grad Kernel", _kernel.shape());
		_saida        = addParam("Saida", shapeOut);
		_buffer 	  = addParam("Buffer", _saida.shape());

		if (usarBias) {
			_bias      = Optional.of(addParam("Bias", shapeOut[0]));
			_gradBias  = Optional.of(addParam("GradBias", _bias.get().shape()));
		}
		
		_treinavel = true;// camada pode ser treinada.
		_construida = true;// camada pode ser usada.
	}

	@Override
	public void inicializar() {
		verificarConstrucao();
		
		iniKernel.forward(_kernel);
		_bias.ifPresent(b -> iniBias.forward(b));
	}

	@Override
	public void setAtivacao(Object act) {
		this.act = new Dicionario().getAtivacao(act);
	}

	@Override
	public void setBias(boolean usarBias) {
		this.usarBias = usarBias;
	}

	@Override
	public void ajustarParaLote(int tamLote) {
		if (tamLote == 0) {
			_gradEntrada = addParam("Grad Entrada", shapeIn);
			_saida = addParam("Saida", shapeOut);
			
		} else {
			final int canais = shapeIn[0];
			final int altIn = shapeIn[1];
			final int largIn = shapeIn[2];

			final int filtros = shapeOut[0];
			final int altOut = shapeOut[1];
			final int largOut = shapeOut[2];

			_gradEntrada = addParam("Grad Entrada", tamLote, canais, altIn, largIn);
			_saida = addParam("Saida", tamLote, filtros, altOut, largOut);
		}

		_buffer = addParam("Buffer", _saida.shape());

		this.tamLote = tamLote;
	}

	/**
	 * <h2>
	 *    Propagação direta através da camada Convolucional
	 * </h2>
	 * <p>
	 *    Realiza a correlação cruzada entre os dados de entrada e os filtros da 
	 *    camada, somando os resultados ponderados. Caso a camada tenha configurado 
	 *    o uso do bias, ele é adicionado após a operação. Por fim é aplicada a função 
	 *    de ativação aos resultados que serão salvos da saída da camada.
	 * </p>
	 * <h3>
	 *    A expressão que define a saída da camada é dada por:
	 * </h3>
	 * <pre>
	 *buffer = corr2D(entrada, kernel);
	 *buffer.add(bias);
	 *saida = ativacao(buffer);
	 * </pre>
	 */
	@Override
	public Tensor forward(Tensor x) {
		verificarConstrucao();

		final int numDim = x.numDim();

		if (numDim == 3) {
			ajustarParaLote(0);
		
		} else if (numDim == 4) {
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

		lops.forwardConv2D(_entrada, _kernel, _bias, _buffer);

		act.forward(_buffer, _saida);

		return _saida;
	}

	/**
	 * <h2>
	 *    Propagação reversa através da camada Convolucional
	 * </h2>
	 * <p>
	 *    Calcula os gradientes da camada para os filtros e bias baseado nos
	 *    gradientes fornecidos.
	 * </p>
	 * <p>
	 *    Após calculdos, os gradientes em relação a entrada da camada são
	 *    calculados e salvos em {@code gradEntrada} para serem retropropagados 
	 *    para as camadas anteriores do modelo em que a camada estiver.
	 * </p>
	 * Resultados calculados ficam salvos nas prorpiedades {@code camada.gradFiltros} e
	 * {@code camada.gradBias}.
	 */
	@Override
	public Tensor backward(Tensor g) {
		verificarConstrucao();

		_gradSaida = g.contiguous();

		act.backward(this, g);

		_gradEntrada.zero();// zerar acumulações anteriores

		lops.backwardConv2D(
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

	/**
	 * Retorna a quantidade de filtros presentes na camada.
	 * @return quantiadde de filtros presentes na camada.
	 */
	public int numFiltros() {
		verificarConstrucao();
		return _kernel.tamDim(0);
	}

	@Override
	public Ativacao ativacao() {
		return act;
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

		int parametros = _kernel.tam();
		if (temBias()) parametros += bias().tam();

		return parametros;
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
		
		sb.append(nome() + " (id " + id + ") = [\n");

		sb.append(pad).append("Entrada: " + JNNutils.arrayStr(shapeIn) + "\n");
		sb.append(pad).append("Filtros: " + numFiltros() + "\n");
		sb.append(pad).append("Saida: " + JNNutils.arrayStr(shapeOut) + "\n");
		sb.append(pad).append("Ativação: " + act.nome() + "\n");
		sb.append("\n");

		sb.append(pad + "Kernel: " + _kernel.shapeStr() + "\n");

		sb.append(pad + "Bias: ");
		if (temBias()) {
			sb.append(_bias.get().shapeStr() + "\n");
		} else {
			sb.append(" N/A\n");
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
	public Conv2D clone() {
		verificarConstrucao();

		Conv2D clone = (Conv2D) super.clone();
		clone.act = this.act;
		clone.usarBias = this.usarBias;
		clone._treinavel = this._treinavel;

		clone._kernel = this._kernel.clone();
		clone._gradKernel = this._gradKernel.clone();
		clone._gradEntrada = this._gradEntrada.clone();

		if (temBias()) {
			clone._bias = Optional.of(bias());
			clone._gradBias = Optional.of(gradBias());
		}

		clone._buffer = this._buffer.clone();
		clone._saida = this._saida.clone();

		return clone;
	}

	/**
	 * Calcula o formato de entrada da camada Convolucional, que é disposto da
	 * seguinte forma:
	 * <pre>
	 *    formato = (profundidade, altura, largura)
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
	 *    formato = (profundidade, altura, largura)
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
	public int[] formatoFiltro() {
		verificarConstrucao();
		return shapeFiltro.clone();
	}

	@Override
	public Tensor kernel() {
		verificarConstrucao();
		return _kernel;
	}

	@Override
	public Tensor gradKernel() {
		verificarConstrucao();
		return _gradKernel;
	}

	@Override
	public Tensor bias() {
		verificarConstrucao();

		return _bias.orElseThrow(() -> new IllegalStateException(
			"\nA camada " + nome() + " (" + id + ") não possui bias configurado."
		));
	}

	@Override
	public Tensor gradBias() {
		verificarConstrucao();

		return _gradBias.orElseThrow(() -> new IllegalStateException(
			"\nA camada " + nome() + " (" + id + ") não possui bias configurado."
		));
	}

	@Override
	public Tensor gradEntrada() {
		verificarConstrucao();
		return _gradEntrada; 
	}

	@Override
	public long tamBytes() {
		String jvmBits = System.getProperty("sun.arch.data.model");
        long bits = Long.valueOf(jvmBits);

        long tamObj;
		// overhead da jvm
        if (bits == 32) tamObj = 8;
        else if (bits == 64) tamObj = 16;
        else throw new IllegalStateException(
            "\nSem suporte para plataforma de " + bits + " bits."
        );

		long tamVars = super.tamBytes(); //base camada
		tamVars += 4 * shapeIn.length; 
		tamVars += 4 * shapeOut.length; 
		tamVars += 4 * shapeFiltro.length; 

		long tamTensores =
		_kernel.tamBytes() + 
		_buffer.tamBytes() +
		_saida.tamBytes() +
		_gradEntrada.tamBytes() +
		_gradKernel.tamBytes();

		if (temBias()) {
			tamTensores += _bias.get().tamBytes();
			tamTensores += _gradBias.get().tamBytes();
		}

		return tamObj + tamVars + tamTensores;
	}

}
