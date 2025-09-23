package jnn.camadas;

import jnn.core.OpTensor;
import jnn.core.Utils;
import jnn.core.tensor.Tensor;
import jnn.core.tensor.Variavel;

/**
 * <h2>
 *    Camada de agrupamento máximo
 * </h2>
 * <p>
 *    A camada de agrupamento máximo é um componente utilizado para reduzir a 
 *    dimensionalidade espacial dos dados, preservando as características mais 
 *    importantes para a saída.
 * </p>
 * <p>
 *    Durante a operação de agrupamento máximo, a entrada é dividida em regiões 
 *    menores usando uma márcara e o valor máximo de cada região é salvo. 
 *    Essencialmente, a camada realiza a operação de subamostragem, mantendo apenas 
 *    as informações mais relevantes.
 * </p>
 * Exemplo simples de operação Max Pooling para uma região 2x2 com máscara 2x2:
 * <pre>
 *entrada = [
 *    [[1, 2],
 *     [3, 4]]
 *]
 * 
 *saida = [
 *    [4]
 *]
 * </pre>
 * <p>
 *    A camada de max pooling não possui parâmetros treináveis nem função de ativação.
 * </p>
 */
public class MaxPool2D extends Camada implements Cloneable{

	/**
	 * Operador para tensores.
	 */
	OpTensor optensor = new OpTensor();

	/**
	 * Utilitario.
	 */
	Utils utils = new Utils();

	/**
	 * Dimensões dos dados de entrada (canais, altura, largura)
	 */
	private int[] shapeEntrada = {1, 1, 1};

	/**
	 * Dimensões dos dados de saída (canais, altura, largura)
	 */
	private int[] shapeSaida = {1, 1, 1};

	/**
	 * Tensor contendo os dados de entrada da camada.
	 * <p>
	 *    O formato da entrada é dado por:
	 * </p>
	 * <pre>
	 *    entrada = (canais, altura, largura)
	 * </pre>
	 */
	public Tensor _entrada;

	/**
	 * Tensor contendo os dados de saída da camada.
	 * <p>
	 *    O formato de entrada varia dependendo da configuração da
	 *    camada (filtro, strides) mas é dado como:
	 * </p>
	 * <pre>
	 *largura = (larguraEntrada = larguraFiltro) / larguraStride + 1;
	 *altura = (alturaEntrada = alturaFiltro) / alturaStride + 1;
	 * </pre>
	 * <p>
	 *    Com isso o formato de saída é dado por:
	 * </p>
	 * <pre>
	 *    saida = (profundidade, altura, largura)
	 * </pre>
	 * Essa relação é válida pra cada canal de entrada.
	 */
	public Tensor _saida;

	/**
	 * Tensor contendo os gradientes que serão
	 * retropropagados para as camadas anteriores.
	 * <p>
	 *    O formato do gradiente de entrada é dado por:
	 * </p>
	 * <pre>
	 *    entrada = (canaisEntrada, alturaEntrada, larguraEntrada)
	 * </pre>
	 */
	public Tensor _gradEntrada;

	/**
	 * Formato do filtro de pooling (altura, largura).
	 */
	private int[] _filtro;

	/**
	 * Valores de stride (altura, largura).
	 */
	private int[] _stride;

	/**
	 * Instancia uma nova camada de max pooling, definindo o formato do
	 * filtro que será aplicado em cada entrada da camada.
	 * <p>
	 *    O formato do filtro deve conter as dimensões da entrada da
	 *    camada (altura, largura).
	 * </p>
	 * <p>
	 *    Por padrão, os valores de strides serão os mesmo usados para
	 *    as dimensões do filtro, exemplo:
	 * </p>
	 * <pre>
	 *filtro = (2, 2)
	 *stride = (2, 2) // valor padrão
	 * </pre>
	 * @param filtro formato do filtro de max pooling.
	 * @throws IllegalArgumentException se o formato do filtro não atender as
	 * requisições.
	 */
	public MaxPool2D(int[] filtro) {
		this(filtro, filtro.clone());
	}

	/**
	 * Instancia uma nova camada de max pooling, definindo o formato do filtro 
	 * e os strides (passos) que serão aplicados em cada entrada da camada.
	 * <p>
	 *    O formato do filtro e dos strides devem conter as dimensões da entrada 
	 *    da camada (altura, largura).
	 * </p>
	 * @param filtro formato do filtro de max pooling.
	 * @param stride strides que serão aplicados ao filtro.
	 * @throws IllegalArgumentException se o formato do filtro não atender as
	 * requisições.
	 * @throws IllegalArgumentException se os strides não atenderem as requisições.
	 */
	public MaxPool2D(int[] filtro, int[] stride) {
		utils.validarNaoNulo(filtro, "\nO formato do filtro não pode ser nulo.");

		if (filtro.length != 2) {
			throw new IllegalArgumentException(
				"\nO formato do filtro deve conter três elementos (altura, largura)."
			);
		}

		if (!utils.apenasMaiorZero(filtro)) {
			throw new IllegalArgumentException(
				"\nOs valores de dimensões do filtro devem ser maiores que zero."
			);
		}

		utils.validarNaoNulo(stride, "\nO formato do filtro não pode ser nulo.");
		
		if (stride.length != 2) {
			throw new IllegalArgumentException(
				"\nO formato para os strides deve conter dois elementos (altura, largura)."
			);
		}

		if (!utils.apenasMaiorZero(stride)) {
			throw new IllegalArgumentException(
				"\nOs valores para os strides devem ser maiores que zero."
			);
		}

		this._filtro = filtro.clone();
		this._stride = stride.clone();
	}

	/**
	 * Instancia uma nova camada de max pooling, definindo o formato do filtro, 
	 * formato de entrada e os strides (passos) que serão aplicados em cada entrada 
	 * da camada.
	 * <p>
	 *    O formato do filtro e dos strides devem conter as dimensões da entrada 
	 *    da camada (altura, largura).
	 * </p>
	 * A camada será automaticamente construída usando o formato de entrada especificado.
	 * @param entrada formato de entrada para a camada.
	 * @param filtro formato do filtro de max pooling.
	 * @param stride strides que serão aplicados ao filtro.
	 * @throws IllegalArgumentException se o formato do filtro não atender as
	 * requisições.
	 * @throws IllegalArgumentException se os strides não atenderem as requisições.
	 */
	public MaxPool2D(int[] entrada, int[] filtro, int[] stride) {
		this(filtro, stride);
		construir(entrada);
	}

	/**
	 * Constroi a camada MaxPooling, inicializando seus atributos.
	 * <p>
	 *    O formato de entrada da camada deve seguir o padrão:
	 * </p>
	 * <pre>
	 *    formEntrada = (profundidade, altura, largura)
	 * </pre>
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

		shapeEntrada[0] = shape[0];// canais
		shapeEntrada[1] = shape[1];// altura
		shapeEntrada[2] = shape[2];// largura

		shapeSaida[0] = shapeEntrada[0];
		shapeSaida[1] = (int) Math.floor((float)(shapeEntrada[1] - _filtro[0]) / _stride[0]) + 1;
		shapeSaida[2] = (int) Math.floor((float)(shapeEntrada[2] - _filtro[1]) / _stride[1]) + 1;

		if (shapeSaida[1] < 1 || shapeSaida[2] < 1) {
			throw new IllegalArgumentException(
				"\nCamada não pode ser construida:" +
				"\nFormato de entrada " + utils.shapeStr(shape) +
				" e formato dos filtros " + utils.shapeStr(_filtro) +
				" resultam num formato de saída inválido " + utils.shapeStr(shapeSaida)
			);
		}
		
		_entrada 	 = addParam("Entrada", shapeEntrada);
		_gradEntrada = addParam("Grad Entrada", _entrada.shape());
		_saida 		 = addParam("Saida", shapeSaida);

		_construida = true;//camada pode ser usada
	}

	@Override
	public void inicializar() {}

	@Override
	public Tensor forward(Object x) {
		verificarConstrucao();

		_entrada.copiar(utils.paraTensor(x));

		optensor.maxPool2D(_entrada, _saida, _filtro, _stride);

		return _saida;
	}

	@Override
	public Tensor backward(Object grad) {
		verificarConstrucao();

		if (grad instanceof Tensor) {
			Tensor g = (Tensor) grad;
			gradMaxPool(_entrada, g, _gradEntrada);
		
		} else {
			throw new IllegalArgumentException(
				"\nTipo de gradiente \"" + grad.getClass().getTypeName() + "\"" +
				" não suportado."
			);
		}

		return _gradEntrada;
	}
	
	/**
	 * Calcula e atualiza os gradientes da camada de Max Pooling em relação à entrada.
	 * <p>
	 *    Retroropaga os gradientes da camada seguinte para a camada de Max Pooling, considerando 
	 *    a operação de agrupamento máximo. Ela calcula os gradientes em relação à entrada para as 
	 *    camadas anteriores.
	 * </p>
	 * @param entrada entrada da camada.
	 * @param gradSeguinte gradiente da camada seguinte.
	 * @param gradEntrada gradiente de entrada da camada de max pooling.
	 */
	private void gradMaxPool(Tensor entrada, Tensor gradSeguinte, Tensor gradEntrada) {
		int[] shapeEntrada = entrada.shape();
		int[] shapeGradS   = gradSeguinte.shape();

		int canais      = shapeEntrada[0];
		int altEntrada  = shapeEntrada[1];
		int largEntrada = shapeEntrada[2];

		int altGradS    = shapeGradS[1];
		int largGradS   = shapeGradS[2];

		// vetorização
		Variavel[] dataE  = entrada.paraArray();
		Variavel[] dataGS = gradSeguinte.paraArray();
		Variavel[] dataGE = gradEntrada.paraArray();

		int canalSizeEntrada = altEntrada * largEntrada;
		int canalSizeGradS   = altGradS * largGradS;
		double val, valMax;

		for (int c = 0; c < canais; c++) {
			int baseEntrada = c * canalSizeEntrada;
			int baseGradS   = c * canalSizeGradS;

			for (int i = 0; i < altGradS; i++) {
				int linInicio = i * _stride[0];
				int linFim    = Math.min(linInicio + _filtro[0], altEntrada);

				for (int j = 0; j < largGradS; j++) {
					int colInicio = j * _stride[1];
					int colFim    = Math.min(colInicio + _filtro[1], largEntrada);

					valMax = Double.NEGATIVE_INFINITY;
					int linMax = linInicio;
					int colMax = colInicio;

					// Encontrar posição do máximo
					for (int y = linInicio; y < linFim; y++) {
						int idLinha = baseEntrada + y * largEntrada;
						for (int x = colInicio; x < colFim; x++) {
							val = dataE[idLinha + x].get();
							if (val > valMax) {
								valMax = val;
								linMax = y;
								colMax = x;
							}
						}
					}

					dataGE[baseEntrada + linMax * largEntrada + colMax].add(dataGS[baseGradS + i * largGradS + j]);
				}
			}
		}

	}
 
	@Override
	public int[] shapeEntrada() {
		verificarConstrucao();
		return shapeEntrada.clone();
	}

	@Override
	public int[] shapeSaida() {
		verificarConstrucao();
		return shapeSaida.clone();
	}

	@Override
	public int tamSaida() {
		verificarConstrucao();
		return _saida.tam();
	}

	/**
	 * Retorna o formato do filtro usado pela camada.
	 * @return dimensões do filtro de pooling.
	 */
	public int[] formatoFiltro() {
		verificarConstrucao();
		return _filtro.clone();
	}
		
	/**
	 * Retorna o formato dos strides usado pela camada.
	 * @return dimensões dos strides.
	 */
	public int[] formatoStride() {
		verificarConstrucao();
		return _stride.clone();
	}

	@Override
	public int numParams() {
		return 0;
	}
	
	@Override
	public Tensor saida() {
		verificarConstrucao();
		return _saida;
	}

	@Override
	public Variavel[] saidaParaArray() {
		return saida().paraArray();
	}

	@Override
	public Tensor gradEntrada() {
		verificarConstrucao();
		return _gradEntrada;
	}

	@Override
	public String info() {
		verificarConstrucao();

		StringBuilder sb = new StringBuilder();
		String pad = " ".repeat(4);
		
		sb.append(nome() + " (id " + this.id + ") = [\n");

		sb.append(pad).append("Entrada: " + utils.shapeStr(shapeEntrada) + "\n");
		sb.append(pad).append("Filtro: " + utils.shapeStr(_filtro) + "\n");
		sb.append(pad).append("Strides: " + utils.shapeStr(_stride) + "\n");
		sb.append(pad).append("Saída: " + utils.shapeStr(shapeSaida()) + "\n");

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
	public MaxPool2D clone() {
		MaxPool2D clone = (MaxPool2D) super.clone();

		clone.optensor = new OpTensor();
		clone.utils = new Utils();

		clone._treinavel = this._treinavel;
		clone.treinando = this.treinando;
		clone._construida = this._construida;

		clone.shapeEntrada = this.shapeEntrada.clone();
		clone._filtro = this._filtro.clone();
		clone.shapeSaida = this.shapeSaida.clone();
		clone._stride = this._stride.clone();
		
		clone._entrada = this._entrada.clone();
		clone._saida = this._saida.clone();
		clone._gradEntrada = this._gradEntrada.clone();

		return clone;
	}

}
