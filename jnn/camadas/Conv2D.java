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
 * <p>
 *    Detalhe adicional:
 * </p>
 * Na realidade a operação que é feita dentro da camada convolucional é chamada de
 * correlação cruzada, é nela que aplicamos os kernels pela entrada recebida. A 
 * verdadeira operação de convolução tem a peculiaridade de rotacionar o filtro 180° 
 * antes de ser executada.
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
	public Tensor _filtros;

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
	public Tensor _bias;

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
	public Tensor _gradFiltros;

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
	public Tensor _gradBias;

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
	 *    formEntrada = (profundidade, altura, largura)
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
	 * @param filtro formato dos filtros da camada.
	 * @param filtros quantidade de filtros.
	 * @param ativacao função de ativação.
	 * @param iniKernel inicializador para os filtros.
	 * @param iniBias inicializador para os bias.
	 */
	public Conv2D(int[] entrada, int[] filtro, int filtros, Object ativacao, Object iniKernel, Object iniBias) {
		this(filtro, filtros, ativacao, iniKernel, iniBias);

		utils.validarNaoNulo(entrada, "\nO formato de entrada não pode ser nulo.");

		if (entrada.length != 3) {
			throw new IllegalArgumentException(
				"\nO formato de entrada deve conter 3 elementos (profundidade, altura, largura), " +
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
	 *    formEntrada = (profundidade, altura, largura)
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
	 * @param filtro formato dos filtros da camada.
	 * @param filtros quantidade de filtros.
	 * @param ativacao função de ativação.
	 * @param iniKernel inicializador para os filtros.
	 */
	public Conv2D(int[] entrada, int[] filtro, int filtros, Object ativacao, Object iniKernel) {
		this(entrada, filtro, filtros, ativacao, iniKernel, null);
	}

	/**
	 * Instancia uma camada convolucional de acordo com os formatos fornecidos.
	 * <p>
	 *    A disposição do formato de entrada deve ser da seguinte forma:
	 * </p>
	 * <pre>
	 *    formEntrada = (profundidade, altura, largura)
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
	 * @param filtro formato dos filtros da camada.
	 * @param filtros quantidade de filtros.
	 * @param ativacao função de ativação.
	 */
	public Conv2D(int[] entrada, int[] filtro, int filtros, Object ativacao) {
		this(entrada, filtro, filtros, ativacao, null, null);
	}

	/**
	 * Instancia uma camada convolucional de acordo com os formatos fornecidos.
	 * <p>
	 *    A disposição do formato de entrada deve ser da seguinte forma:
	 * </p>
	 * <pre>
	 *    formEntrada = (altura, largura, profundidade)
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
	 * @param filtro formato dos filtros da camada.
	 * @param filtros quantidade de filtros.
	 */
	public Conv2D(int[] entrada, int[] filtro, int filtros) {
		this(entrada, filtro, filtros, null, null, null);
	}

	/**
	 * Instancia uma camada convolucional de acordo com os formatos fornecidos.
	 * <p>
	 *    A disposição do formato de entrada deve ser da seguinte forma:
	 * </p>
	 * <pre>
	 *    formEntrada = (profundidade, altura, largura)
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
	public Conv2D(int[] filtro, int filtros, Object ativacao, Object iniKernel, Object iniBias) {
		utils.validarNaoNulo(filtro, "\nO formato do filtro não pode ser nulo.");

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

		//número de filtros
		if (filtros < 1) {
			throw new IllegalArgumentException(
				"\nO número de filtro deve ser maior que zero, recebido: " + filtros
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
	 *    formEntrada = (profundidade, altura, largura)
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
	public Conv2D(int[] filtro, int filtros, Object ativacao, Object iniKernel) {
		this(filtro, filtros, ativacao, iniKernel, null);
	}

	/**
	 * Instancia uma camada convolucional de acordo com os formatos fornecidos.
	 * <p>
	 *    A disposição do formato de entrada deve ser da seguinte forma:
	 * </p>
	 * <pre>
	 *    formEntrada = (profundidade, altura, largura)
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
	public Conv2D(int[] filtro, int filtros, Object ativacao) {
		this(filtro, filtros, ativacao, null, null);
	}

	/**
	 * Instancia uma camada convolucional de acordo com os formatos fornecidos.
	 * <p>
	 *    A disposição do formato de entrada deve ser da seguinte forma:
	 * </p>
	 * <pre>
	 *    formEntrada = (profundidade, altura, largura)
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
	public Conv2D(int[] filtro, int filtros) {
		this(filtro, filtros, null, null, null);
	}
	
	/**
	 * Inicializa os parâmetros necessários para a camada Convolucional.
	 * <p>
	 *    O formato de entrada deve ser um array contendo o tamanho de 
	 *    cada dimensão de entrada da camada, e deve estar no formato:
	 * </p>
	 * <pre>
	 *    entrada = (profundidade, altura, largura)
	 * </pre>
	 * @param entrada formato de entrada para a camada.
	 */
	@Override
	public void construir(Object entrada) {
		utils.validarNaoNulo(entrada, "Formato de entrada fornecida para camada Convolucional é nulo.");

		if (!(entrada instanceof int[])) {
			throw new IllegalArgumentException(
				"\nObjeto esperado para entrada da camada Convolucional é do tipo int[], " +
				"objeto recebido é do tipo " + entrada.getClass().getTypeName()
			);
		}

		int[] fEntrada = (int[]) entrada;

		if (fEntrada.length != 3 && fEntrada.length != 4) {
			throw new IllegalArgumentException(
				"\nO formato de entrada para a camada Convolucional deve conter três " + 
				"elementos (profundidade, altura, largura), ou quatro elementos (primeiro desconsiderado) " + 
				"objeto recebido possui " + fEntrada.length
			);
		}

		shapeEntrada[0] = (fEntrada.length == 4) ? fEntrada[1] : fEntrada[0];//profundidade
		shapeEntrada[1] = (fEntrada.length == 4) ? fEntrada[2] : fEntrada[1];//altura
		shapeEntrada[2] = (fEntrada.length == 4) ? fEntrada[3] : fEntrada[2];//largura

		if (!utils.apenasMaiorZero(fEntrada)) {
			throw new IllegalArgumentException(
				"\nOs valores de dimensões de entrada para a camada Convolucional não " +
				"podem conter valores menores que 1."
			);
		}

		//dim -> ((entrada - filtro) / stride) + 1
		shapeSaida[1] = shapeEntrada[1] - shapeFiltro[0] + 1;
		shapeSaida[2] = shapeEntrada[2] - shapeFiltro[1] + 1;

		if (shapeSaida[1] < 1 || shapeSaida[2] < 1) {
			throw new IllegalArgumentException(
				"\nFormato de entrada " + utils.shapeStr(fEntrada) +
				" e formato dos filtros " + 
				utils.shapeStr(new int[]{shapeSaida[0], shapeFiltro[0], shapeFiltro[1]}) +
				" resultam num formato de saída inválido " + utils.shapeStr(shapeSaida)
			);
		}

		//inicialização dos parâmetros necessários
		_entrada      = new Tensor(shapeEntrada);
		_gradEntrada  = new Tensor(_entrada.shape());
		_filtros      = new Tensor(shapeSaida[0], shapeEntrada[0], shapeFiltro[0], shapeFiltro[1]);
		_gradFiltros  = new Tensor(_filtros.shape());
		_saida        = new Tensor(shapeSaida);
		_somatorio    = new Tensor(_saida.shape());
		_gradSaida    = new Tensor(_saida.shape());

		if (usarBias) {
			_bias      = new Tensor(shapeSaida[0]);
			_gradBias  = new Tensor(_bias.shape());
		}

		setNomes();
		
		_treinavel = true;
		_construida = true;//camada pode ser usada.
	}

	@Override
	public void inicializar() {
		verificarConstrucao();
		
		iniKernel.inicializar(_filtros);

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
		_gradEntrada.nome("gradiente entrada");
		_filtros.nome("kernel");
		_saida.nome("saida");
		_gradFiltros.nome("gradiente kernel");
		_somatorio.nome("somatório");
		_gradSaida.nome("gradiente saída");

		if (usarBias) {
			_bias.nome("bias");
			_gradBias.nome("gradiente bias");
		}
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
			Tensor e = (Tensor) entrada;
			if (!_entrada.compararShape(e)) {
				throw new IllegalArgumentException(
					"\nAs dimensões da entrada recebida " + e.shapeStr() + 
					" são incompatíveis com as dimensões da entrada da camada " + _entrada.shapeStr()
				);
			}

			_entrada.copiar(e);
		
		} else if (entrada instanceof double[][][]) {
			double[][][] e = (double[][][]) entrada;
			_entrada.copiar(e);
		
		} else {
			throw new IllegalArgumentException(
				"\nOs dados de entrada para a camada Convolucional devem ser " +
				"do tipo " + _entrada.getClass().getSimpleName() + 
				" ou double[][][] objeto recebido é do tipo \"" + 
				entrada.getClass().getTypeName() + "\"."
			);
		}

		//feedforward

		//zerar os valores calculados anteiormente
		_somatorio.preencher(0.0d);

		optensor.conv2DForward(_entrada, _filtros, _somatorio);
		
		// TODO melhorar isso usando broadcasting de tensores
		if (usarBias) {
			int f = numFiltros();
			int alt = shapeSaida[1];
			int larg = shapeSaida[2];
			for (int i = 0; i < f; i++) {
				double b = _bias.get(i);
				for (int j = 0; j < alt; j++) {
					for (int k = 0; k < larg; k++) {
						_somatorio.add(b, i, j, k);
					}
				}
			}
		}

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
			Tensor g = (Tensor) grad;
			if (!_gradSaida.compararShape(g)) {
				throw new IllegalArgumentException(
					"\nGradiente deve ter formato " + _gradSaida.shapeStr() +
					" mas tem " + g.shapeStr()
				);
			}

			_gradSaida.copiar(g);

		} else {
			throw new IllegalArgumentException(
				"Os gradientes para a camada Convolucional devem ser " +
				"do tipo \"" + _gradSaida.getClass().getTypeName() + 
				"\", objeto recebido é do tipo \"" + grad.getClass().getTypeName() + "\""
			);
		}

		ativacao.backward(this);
		
		//backward
		Tensor tempGrad = new Tensor(_gradFiltros.shape());
		_gradEntrada.preencher(0.0d);
		
		optensor.conv2DBackward(_entrada, _filtros, _gradSaida, tempGrad, _gradEntrada);
		_gradFiltros.add(tempGrad);

		if (usarBias) {
			//TODO melhorar isso usando broadcasting, se der
			final int f = numFiltros();
			final int alt = shapeSaida[1];
			final int larg = shapeSaida[2];
			for (int i = 0; i < f; i++) {
				double soma = 0.0;
				for (int j = 0; j < alt; j++) {
					for (int k = 0; k < larg; k++) {
						soma += _gradSaida.get(i, j, k);
					}
				}
				_gradBias.add(soma, i);
			}
		}

		return _gradEntrada;
	}

	@Override
	public void zerarGradientes() {
		verificarConstrucao();

		_gradFiltros.zerar();
		if (usarBias) _gradBias.zerar();
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
		return _saida;
	}

	@Override
	public boolean temBias() {
		return usarBias;
	}

	@Override
	public int numParametros() {
		verificarConstrucao();

		int parametros = _filtros.tamanho();
		
		if (usarBias) parametros += _bias.tamanho();

		return parametros;
	}

	@Override
	public Variavel[] saidaParaArray() {
		verificarConstrucao();

		return _saida.paraArray();
	}

	@Override 
	public int tamanhoSaida() {
		return _saida.tamanho();
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

		sb.append(pad + "Kernel: " + _filtros.shapeStr() + "\n");

		sb.append(pad + "Bias: ");
		if (temBias()) {
			sb.append(_bias.shapeStr() + "\n");
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
		clone._filtros     = this._filtros.clone();
		clone._gradFiltros = this._gradFiltros.clone();
		clone._gradEntrada = this._gradEntrada.clone();

		if (this.usarBias) {
			clone._bias     = this._bias.clone();
			clone._gradBias = this._gradBias.clone();
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
	public int[] formatoEntrada() {
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
	public int[] formatoSaida() {
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
		return _filtros;
	}

	@Override
	public Variavel[] kernelParaArray() {
		return kernel().paraArray();
	}

	@Override
	public Tensor gradKernel() {
		return _gradFiltros;
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
	public void setKernel(Variavel[] kernel) {
		if (kernel.length != _filtros.tamanho()) {
			throw new IllegalArgumentException(
				"A dimensão do kernel fornecido (" + kernel.length + ") não é igual a quantidade de " +
				" parâmetros para os kernels da camada (" + _filtros.tamanho() + ")."
			);
		}
			
		_filtros.copiarElementos(kernel);
	}

	@Override
	public void setBias(Variavel[] bias) {
		if (bias.length != _bias.tamanho()) {
			throw new IllegalArgumentException(
				"A dimensão do bias fornecido não é igual a quantidade de " +
				" parâmetros para os bias da camada."
			);
		}
		
		_bias.copiarElementos(bias);
	}

}
