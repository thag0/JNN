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
public class Densa extends Camada implements Cloneable {

	/**
	 * Utilitário.
	 */
	private LayerOps lops = new LayerOps();

	/**
	 * Variável controlador para o tamanho de entrada da camada densa.
	 */
	private int[] shapeIn = {1};
	 
	/**
	 * Variável controlador para a quantidade de neurônios (unidades) 
	 * da camada densa.
	 */
	private int[] shapeOut = {1};

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
	private boolean usarBias = true;

	/**
	 * Tensor contendo os valores de entrada da camada, seu formato se dá por:
	 * <pre>
	 *    entrada = (tamEntrada)
	 * </pre>
	 */
	public Tensor _entrada;

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
	 * @param iniKernel inicializador para os pesos da camada.
	 * @param iniBias inicializador para os bias da camada.
	 */
	public Densa(int e, int n, Object iniKernel, Object iniBias) {
		this(n, iniKernel, iniBias);
		construir(new int[]{ e });// construir automaticamente
	}

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param e quantidade de conexões de entrada.
	 * @param n quantidade de neurônios (unidades).
	 * @param iniKernel inicializador para os pesos da camada.
	 */
	public Densa(int e, int n, Object iniKernel) {
		this(e, n, iniKernel, null);
	}

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param e quantidade de conexões de entrada.
	 * @param n quantidade de neurônios (unidades).
	 */
	public Densa(int e, int n) {
		this(e, n, null, null);
	}

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param n quantidade de neurônios (unidades).
	 * @param iniKernel inicializador para os pesos da camada.
	 * @param iniBias inicializador para os bias da camada.
	 */
	public Densa(int n, Object iniKernel, Object iniBias) {
		if (n < 1) {
			throw new IllegalArgumentException(
				"\nA camada deve conter ao menos um neurônio."
			);
		}

		shapeOut[0] = n;

		//usar os valores padrão se necessário
		Dicionario dic = new Dicionario();
		if (iniKernel != null) this.iniKernel = dic.getInicializador(iniKernel);
		if (iniBias != null)  this.iniBias = dic.getInicializador(iniBias);
	}

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param n quantidade de neurônios (unidades).
	 * @param iniKernel inicializador para os pesos da camada.
	 */
	public Densa(int n, Object iniKernel) {
		this(n, iniKernel, null);
	}

	/**
	 * Instancia uma nova camada densa de neurônios de acordo com os dados fornecidos.
	 * Após a inicialização os pesos e bias da Camada estarão zerados e devem ser 
	 * inicializados com o método {@code inicializar()}.
	 * @param n quantidade de neurônios (unidades).
	 */
	public Densa(int n) {
		this(n, null, null);
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
		
		if (usarBias) _bias = Optional.of(addParam("bias", shapeOut[0]));
		else _bias = Optional.empty();

		_gradEntrada = addBuffer("Grad Entrada", shapeIn[0]);
		_saida  	 = addBuffer("Saida", shapeOut[0]);
		
		_treinavel = true;// camada pode ser treinada.
		construida = true;// camada pode ser usada.
	}

	@Override
	public void initParams() {
		verificarConstrucao();

		iniKernel.forward(_kernel.weight);
		_bias.ifPresent(b -> iniBias.forward(b.weight));
	}

	@Override
	public void setBias(boolean usarBias) {
		this.usarBias = usarBias;
	}

	@Override
	public void ajustarParaLote(int tamLote) {
		int[] shapeIn, out;

		if (tamLote == 0) {
			shapeIn = new int[]{ tamEntrada() };
			out = shapeOut;
			
		} else {
			shapeIn = new int[]{ tamLote, tamEntrada() };
			out = new int[]{ tamLote, numNeuronios() };
		}

		_gradEntrada = addBuffer("Grad Entrada", shapeIn);
		_saida 		 = addBuffer("Saida", out);
		
		this.tamLote = tamLote;
	}

	@Override
	public Tensor forward(Tensor x) {
		verificarConstrucao();

		if (x.numDim() == 1) {
			validarShapes(x.shape(), shapeIn);
			ajustarParaLote(0);
			
		} else if (x.numDim() == 2) {
			validarShapes(x.shape(), shapeIn);
			int lotes = x.tamDim(0);
			if (this.tamLote != lotes) {
				ajustarParaLote(lotes);
			}
			
		} else {
			throw new UnsupportedOperationException(
				"\n Dimensões de X " + x.numDim() + " não suportadas."
			);
		}

		_entrada = x.contiguous();

		lops.forwardDensa(_entrada, _kernel, _bias, _saida);

		return _saida;
	}

	@Override
	public Tensor backward(Tensor g) {
		verificarConstrucao();
		
		_gradSaida = g.contiguous();

		lops.backwardDensa(
			_entrada,
			_kernel,
			_gradSaida,
			_bias,
			_gradEntrada
		);

		return _gradEntrada;
	}

	@Override
	public void gradZero() {
		verificarConstrucao();

		_kernel.grad.zero();
		_bias.ifPresent(b -> b.grad.zero());
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
		return shapeOut[0];
	}

	/**
	 * Retorna a capacidade de entrada da camada.
	 * @return tamanho de entrada da camada.
	 */
	public int tamEntrada() {
		verificarConstrucao();
		return shapeIn[0];
	}

	/**
	 * Retorna a capacidade de saída da camada.
	 * @return tamanho de saída da camada.
	 */
	public int tamSaida() {
		verificarConstrucao();
		return numNeuronios();
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
	public String info() {
		verificarConstrucao();

		StringBuilder sb = new StringBuilder();
		String pad = " ".repeat(4);
		
		sb.append(nome()).append("(\n");

		sb.append(pad).append("In: ").append(JNNutils.arrayStr(shapeIn)).append("\n");
		sb.append(pad).append("Out: ").append(JNNutils.arrayStr(shapeOut)).append("\n");

		sb.append(pad).append(_kernel).append("\n");
		if (temBias()) sb.append(pad).append(_bias.get()).append("\n");

		sb.append(")");

		return sb.toString();
	}

	@Override
	public Densa clone() {
		verificarConstrucao();

		Densa clone = (Densa) super.clone();

		clone.lops = new LayerOps();
		clone._treinavel = this._treinavel;

		clone._kernel = _kernel.clone();
		clone._kernel.grad.copiar(_kernel.grad);

		clone.usarBias = this.usarBias;
		if (temBias()) {
			clone._bias = Optional.of(_bias.get().clone());
			clone._bias.get().grad.copiar(_bias.get().grad);
		}

		clone._saida = this._saida.clone();

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
		verificarConstrucao();
		return _gradEntrada.shape();
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
		verificarConstrucao();
		return _saida.shape();
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
		long tamVars = super.tamBytes(); //base camada + tensores
		tamVars += 4; //tamEntrada
		tamVars += 4; //numNeuronios
		tamVars += 1; //usarBias

		long tamTensores =
		_saida.tamBytes() +
		_gradEntrada.tamBytes();

		return tamVars + tamTensores;
	}

}
