package jnn.camadas;

import java.util.Optional;

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
 *    Camada Convolucional
 * </h2>
 * <p>
 *    A camada convolucional realiza operações de convolução sobre a entrada
 *    utilizando filtros (kernels) para extrair características locais, dada 
 *    pela expressão:.
 * </p>
 * <pre>
 *    somatorio = convolucao(entrada, filtros) + bias
 * </pre>
 * Após a propagação dos dados, a função de ativação da camada é aplicada ao 
 * resultado do somatório, que por fim é salvo na saída da camada.
 * <pre>
 *    saida = ativacao(somatorio)
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
	 * Operador de tensores para a camada.
	 */
	private OpTensor optensor = new OpTensor();

	/**
	 * Utilitário.
	 */
	private Utils utils = new Utils();

	/**
	 * Formato de entrada da camada convolucional, dado por:
	 * <pre>
	 *    form = (profundidade, altura, largura)
	 * </pre>
	 */
	private final int[] shapeEntrada = {1, 1, 1};

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
	 *    form = (numFiltros, altura, largura)
	 * </pre>
	 */
	private final int[] shapeSaida = {1, 1, 1};

	/**
	 * Tensor contendo os valores de entrada para a camada,
	 * que serão usados para o processo de feedforward.
	 * <p>
	 *    O formato da entrada é dado por:
	 * </p>
	 * <pre>
	 *    entrada = (profundidade, altura, largura)
	 * </pre>
	 */
	public Tensor _entrada;

	/**
	 * Tensor contendo os filtros (ou kernels)
	 * da camada.
	 * <p>
	 *    O formato dos filtros é dado por:
	 * </p>
	 * <pre>
	 *    entrada = (numFiltros, profundidadeEntrada, alturaFiltro, larguraFiltro)
	 * </pre>
	 */
	public Tensor _kernel;

	/**
	 * Tensor contendo os bias (vieses) para cada valor de 
	 * saída da camada.
	 * <p>
	 *    O formato do bias é dado por:
	 * </p>
	 * <pre>
	 *    bias = (numFiltros)
	 * </pre>
	 */
	public Optional<Tensor> _bias;

	/**
	 * Auxiliar na verificação de uso do bias.
	 */
	private boolean usarBias = true;

	/**
	 * Tensor contendo valores resultantes do cálculo de correlação cruzada
	 * entre a entrada e os filtros, com o bias adicionado (se houver).
	 * <p>
	 *    O formato somatório é dado por:
	 * </p>
	 * <pre>
	 *    somatorio = (numeroFiltros, alturaSaida, larguraSaida)
	 * </pre>
	 */
	public Tensor _somatorio;
	
	/**
	 * Tensor contendo os valores de saídas da camada.
	 * <p>
	 *    O formato da saída é dado por:
	 * </p>
	 * <pre>
	 *    saida = (numeroFiltros, alturaSaida, larguraSaida)
	 * </pre>
	 */
	public Tensor _saida;

	/**
	 * Tensor contendo os valores dos gradientes usados para 
	 * a retropropagação para camadas anteriores.
	 * <p>
	 *    O formato do gradiente de entrada é dado por:
	 * </p>
	 * <pre>
	 *    gradEntrada = (profEntrada, alturaEntrada, larguraEntrada)
	 * </pre>
	 */
	public Tensor _gradEntrada;

	/**
	 * Tensor contendo os valores dos gradientes relativos a saída
	 * da camada.
	 * <p>
	 *    O formato dos gradientes da saída é dado por:
	 * </p>
	 * <pre>
	 *    gradSaida = (numFiltros, alturaSaida, larguraSaida)
	 * </pre>
	 */
	public Tensor _gradSaida;

	/**
	 * Tensor contendo os valores dos gradientes relativos a cada
	 * filtro da camada.
	 * <p>
	 *    O formato dos gradientes para os filtros é dado por:
	 * </p>
	 * <pre>
	 * gradFiltros = (numFiltros, profundidadeEntrada, alturaFiltro, larguraFiltro)
	 * </pre>
	 */
	public Tensor _gradKernel;

	/**
	 * Tensor contendo os valores dos gradientes relativos a cada
	 * bias da camada.
	 * <p>
	 *    O formato dos gradientes para os bias é dado por:
	 * </p>
	 * <pre>
	 *    gradBias = (numFiltros)
	 * </pre>
	 */
	public Optional<Tensor> _gradBias;

	/**
	 * Função de ativação da camada.
	 */
	private Ativacao ativacao = new Linear();

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
	 * @param ativacao função de ativação.
	 * @param iniKernel inicializador para os filtros.
	 * @param iniBias inicializador para os bias.
	 */
	public Conv2D(int[] entrada, int filtros, int[] filtro, Object ativacao, Object iniKernel, Object iniBias) {
		this(filtros, filtro, ativacao, iniKernel, iniBias);

		utils.validarNaoNulo(entrada, "O formato de entrada nulo.");

		if (entrada.length != 3) {
			throw new IllegalArgumentException(
				"\nO formato de entrada deve conter 3 elementos (canais, altura, largura), " +
				"recebido: " + entrada.length
			);
		}

		if (!utils.apenasMaiorZero(entrada)) {
			throw new IllegalArgumentException(
				"\nOs valores do formato de entrada devem ser maiores que zero."
			);
		}

		shapeEntrada[0] = entrada[0];//profundidade
		shapeEntrada[1] = entrada[1];//altura
		shapeEntrada[2] = entrada[2];//largura

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
	 * @param ativacao função de ativação.
	 * @param iniKernel inicializador para os filtros.
	 */
	public Conv2D(int[] entrada, int filtros, int[] filtro, Object ativacao, Object iniKernel) {
		this(entrada, filtros, filtro, ativacao, iniKernel, null);
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
	 * @param ativacao função de ativação.
	 */
	public Conv2D(int[] entrada, int filtros, int[] filtro, Object ativacao) {
		this(entrada, filtros, filtro, ativacao, null, null);
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
	 * @param ativacao função de ativação.
	 * @param iniKernel inicializador para os filtros.
	 * @param iniBias inicializador para os bias.
	 */
	public Conv2D(int filtros, int[] filtro, Object ativacao, Object iniKernel, Object iniBias) {
		utils.validarNaoNulo(filtro, "O formato do filtro não pode ser nulo.");

		//formado dos filtros
		int[] formFiltro = (int[]) filtro;
		if (formFiltro.length != 2) {
			throw new IllegalArgumentException(
				"\nO formato dos filtros deve conter 2 elementos (altura, largura), " +
				"recebido: " + formFiltro.length
			);
		}

		if (!utils.apenasMaiorZero(formFiltro)) {
			throw new IllegalArgumentException(
				"\nOs valores de formato para os filtros devem ser maiores que zero."
			);      
		}

		shapeFiltro[0] = formFiltro[0];
		shapeFiltro[1] = formFiltro[1];

		// número de filtros
		if (filtros < 1) {
			throw new IllegalArgumentException(
				"\nO número de filtros deve ser maior que zero, recebido: " + filtros
			);
		}

		shapeSaida[0] = filtros;
		
		Dicionario dicio = new Dicionario();
		if (ativacao != null) this.ativacao = dicio.getAtivacao(ativacao);
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
	 * @param ativacao função de ativação.
	 * @param iniKernel inicializador para os filtros.
	 */
	public Conv2D(int filtros, int[] filtro, Object ativacao, Object iniKernel) {
		this(filtros, filtro, ativacao, iniKernel, null);
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
	 * @param ativacao função de ativação.
	 */
	public Conv2D(int filtros, int[] filtro, Object ativacao) {
		this(filtros, filtro, ativacao, null, null);
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
		utils.validarNaoNulo(shape, "Formato de entrada nulo.");

		if (shape.length != 3) {
			throw new IllegalArgumentException(
				"\nFormato de entrada para a camada " + nome() + " deve conter três " + 
				"elementos (canais, altura, largura), mas recebido tamanho = " + shape.length
			);
		}

		if (!utils.apenasMaiorZero(shape)) {
			throw new IllegalArgumentException(
				"\nOs valores de dimensões de entrada para a camada " + nome() + 
				" devem ser maiores que zero."
			);
		}

		shapeEntrada[0] = shape[0];// canais
		shapeEntrada[1] = shape[1];// altura
		shapeEntrada[2] = shape[2];// largura

		//dim -> ((entrada - filtro) / stride) + 1
		shapeSaida[1] = shapeEntrada[1] - shapeFiltro[0] + 1;
		shapeSaida[2] = shapeEntrada[2] - shapeFiltro[1] + 1;

		if (shapeSaida[1] < 1 || shapeSaida[2] < 1) {
			throw new IllegalArgumentException(
				"\nFormato de entrada " + utils.shapeStr(shape) +
				" e formato dos filtros " + 
				utils.shapeStr(new int[]{shapeSaida[0], shapeFiltro[0], shapeFiltro[1]}) +
				" resultam num formato de saída inválido " + utils.shapeStr(shapeSaida)
			);
		}

		//inicialização dos parâmetros necessários
		_entrada      = new Tensor(shapeEntrada);
		_gradEntrada  = new Tensor(_entrada.shape());
		_kernel       = new Tensor(shapeSaida[0], shapeEntrada[0], shapeFiltro[0], shapeFiltro[1]);
		_gradKernel   = new Tensor(_kernel.shape());
		_saida        = new Tensor(shapeSaida);
		_somatorio    = new Tensor(_saida.shape());
		_gradSaida    = new Tensor(_saida.shape());

		if (usarBias) {
			_bias      = Optional.of(new Tensor(shapeSaida[0]));
			_gradBias  = Optional.of(new Tensor(shapeSaida[0]));
		}

		setNomes();
		
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
		
		iniKernel.inicializar(_kernel);
		_bias.ifPresent(b -> iniBias.inicializar(b));
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
		_kernel.nome("kernel");
		_saida.nome("saida");
		_somatorio.nome("somatório");
		_gradEntrada.nome("grad entrada");
		_gradKernel.nome("grad kernel");
		_gradSaida.nome("grad saída");

		_bias.ifPresent(b -> b.nome("bias"));
		_gradBias.ifPresent(gb -> gb.nome("grad bias"));
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
	 *somatorio = correlacaoCruzada(entrada, filtros)
	 *somatorio.add(bias)
	 *saida = ativacao(somatorio)
	 * </pre>
	 * @param entrada dados de entrada que serão processados, tipos aceitos são,
	 * {@code double[][][]} ou {@code Tensor}.
	 */
	@Override
	public Tensor forward(Object entrada) {
		verificarConstrucao();

		if (entrada instanceof Tensor) {
			_entrada.copiar((Tensor) entrada);
		
		} else if (entrada instanceof double[][][]) {
			_entrada.copiar((double[][][]) entrada);
		
		} else {
			throw new IllegalArgumentException(
				"\nTipo de entrada \"" + entrada.getClass().getTypeName() + "\"" +
				" não suportada."
			);
		}
		
		_somatorio.zerar();// zerar valores pre-calculados
		optensor.conv2DForward(_entrada, _kernel, _bias, _somatorio);

		ativacao.forward(_somatorio, _saida);

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
	 * @param grad gradiente da camada seguinte.
	 */
	@Override
	public Tensor backward(Object grad) {
		verificarConstrucao();

		if (grad instanceof Tensor) {
			_gradSaida.copiar((Tensor) grad);

		} else if (grad instanceof double[][][]) {
			_gradSaida.copiar((double[][][]) grad);

		} else {
			throw new IllegalArgumentException(
				"\nTipo de gradiente \"" + grad.getClass().getTypeName() + "\"" +
				" não suportado."
			);
		}

		ativacao.backward(this);

		_gradEntrada.zerar();
		Tensor temp = new Tensor(_gradKernel.shape());
		
		optensor.conv2DBackward(_entrada, _kernel, _gradSaida, temp, _gradBias, _gradEntrada);
		_gradKernel.add(temp);

		return _gradEntrada;
	}

	@Override
	public void zerarGrad() {
		verificarConstrucao();

		_gradKernel.zerar();
		_bias.ifPresent(b -> b.zerar());
	}

	/**
	 * Retorna a quantidade de filtros presentes na camada.
	 * @return quantiadde de filtros presentes na camada.
	 */
	public int numFiltros() {
		verificarConstrucao();
		return this.shapeSaida[0];
	}

	@Override
	public Ativacao ativacao() {
		return ativacao;
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
	public int numParametros() {
		verificarConstrucao();

		int parametros = _kernel.tamanho();
		if (temBias()) parametros += bias().tamanho();

		return parametros;
	}

	@Override
	public Variavel[] saidaParaArray() {
		return saida().paraArray();
	}

	@Override 
	public int tamanhoSaida() {
		return saida().tamanho();
	}

	@Override
	public String info() {
		verificarConstrucao();

		StringBuilder sb = new StringBuilder();
		String pad = " ".repeat(4);
		
		sb.append(nome() + " (id " + id + ") = [\n");

		sb.append(pad).append("Ativação: " + ativacao.nome() + "\n");
		sb.append(pad).append("Entrada: " + utils.shapeStr(shapeEntrada) + "\n");
		sb.append(pad).append("Filtros: " + numFiltros() + "\n");
		sb.append(pad).append("Saida: " + utils.shapeStr(shapeSaida) + "\n");
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
		clone.ativacao 	 = this.ativacao;
		clone.usarBias   = this.usarBias;
		clone._treinavel = this._treinavel;

		clone._entrada     = this._entrada.clone();
		clone._kernel     = this._kernel.clone();
		clone._gradKernel = this._gradKernel.clone();
		clone._gradEntrada = this._gradEntrada.clone();

		if (temBias()) {
			clone._bias 	= Optional.of(bias());
			clone._gradBias = Optional.of(gradBias());
		}

		clone._somatorio   = this._somatorio.clone();
		clone._saida       = this._saida.clone();
		clone._gradSaida   = this._gradSaida.clone();

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
	public int[] shapeEntrada() {
		verificarConstrucao();
		return shapeEntrada.clone();
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
	public int[] shapeSaida() {
		verificarConstrucao();
		return shapeSaida.clone();
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
