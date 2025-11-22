package jnn.camadas;

import jnn.core.Utils;
import jnn.core.tensor.Tensor;

/**
 * <h1>
 *    Camada de agrupamento médio
 * </h1>
 * <p>
 *    A camada de agrupamento médio é um componente utilizado para reduzir a 
 *    dimensionalidade espacial dos dados, preservando as características mais 
 *    importantes para a saída.
 * </p>
 * <p>
 *    Durante a operação de agrupamento médio, a entrada é dividida em regiões 
 *    menores usando uma máscara e a média de cada região é calculada e salva. 
 *    Essencialmente, a camada realiza a operação de subamostragem, calculando a 
 *    média das informações em cada região.
 * </p>
 * Exemplo simples de operação Avg Pooling para uma região 2x2 com máscara 2x2:
 * <pre>
 *entrada = [
 *    [[1, 2],
 *     [3, 4]]
 *]
 * 
 *saida = [2.5]
 * </pre>
 * <p>
 *    A camada de avg pooling não possui parâmetros treináveis nem função de ativação.
 * </p>
 */
public class AvgPool2D extends Camada {

	/**
	 * Utilitário.
	 */
	LayerOps lops = new LayerOps();

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
	 * Auxilar no controle de treinamento em lotes.
	 */
	private int tamLote;

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
	 *    saida = (canais, altura, largura)
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
	 *    entrada = (canaisEntrada, alturaEntrada, larguraEntrad)
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
	 * Instancia uma nova camada de average pooling, definindo o formato do filtro 
	 * e os strides (passos) que serão aplicados em cada entrada da camada.
	 * <p>
	 *    O formato do filtro e dos strides devem conter as dimensões da entrada 
	 *    da camada (altura, largura).
	 * </p>
	 * @param filtro formato do filtro de average pooling.
	 * @param stride strides que serão aplicados ao filtro.
	 */
	public AvgPool2D(int[] filtro, int[] stride) {
		utils.validarNaoNulo(filtro, "Formato do filtro nulo.");

		if (filtro.length != 2) {
			throw new IllegalArgumentException(
				"\nO formato do filtro deve conter dois elementos (altura, largura)."
			);
		}

		if (!utils.apenasMaiorZero(filtro)) {
			throw new IllegalArgumentException(
				"\nOs valores de dimensões do filtro devem ser maiores que zero."
			);
		}

		utils.validarNaoNulo(stride, "Formato de stride nulo.");

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
	 * Instancia uma nova camada de average pooling, definindo o formato do
	 * filtro que será aplicado em cada entrada da camada.
	 * <p>
	 *    O formato do filtro deve conter as dimensões da entrada da
	 *    camada (altura, largura).
	 * </p>
	 * <p>
	 *    Por padrão, os valores de strides serão os mesmos usados para
	 *    as dimensões do filtro, exemplo:
	 * </p>
	 * <pre>
	 *filtro = (2, 2)
	 *stride = (2, 2) // valor padrão
	 * </pre>
	 * @param filtro formato do filtro de average pooling.
	 */
	public AvgPool2D(int[] filtro) {
		this(filtro, filtro.clone());
	}

	/**
	 * Instancia uma nova camada de average pooling, definindo o formato do filtro, 
	 * formato de entrada e os strides (passos) que serão aplicados em cada entrada 
	 * da camada.
	 * <p>
	 *    O formato do filtro e dos strides devem conter as dimensões da entrada 
	 *    da camada (altura, largura).
	 * </p>
	 * A camada será automaticamente construída usando o formato de entrada especificado.
	 * @param entrada formato de entrada para a camada.
	 * @param filtro formato do filtro de average pooling.
	 * @param stride strides que serão aplicados ao filtro.
	 */
	public AvgPool2D(int[] entrada, int[] filtro, int[] stride) {
		this(filtro, stride);
		construir(entrada);
	}

	/**
	 * Constroi a camada AvgPooling, inicializando seus atributos.
	 * <p>
	 *    O formato de entrada da camada deve seguir o padrão:
	 * </p>
	 * <pre>
	 *    formEntrada = (canais, altura, largura)
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

		_construida = true;// camada pode ser usada
	}

	@Override
	public void inicializar() {}

	@Override
	public void ajustarParaLote(int tamLote) {
		final int canais = shapeEntrada[0];
		final int altIn = shapeEntrada[1];
		final int largIn = shapeEntrada[2];
		
		final int altOut = shapeSaida[1];
		final int largOut = shapeSaida[2];

		if (tamLote == 0) {
			_entrada = addParam("Entrada", shapeEntrada);
			_saida = addParam("Saida", shapeSaida);
			
		} else {
			_entrada = addParam("Entrada", tamLote, canais, altIn, largIn);
			_saida = addParam("Saida", tamLote, canais, altOut, largOut);
		}

		_gradEntrada = addParam("Grad Entrada", _entrada.shape());

		this.tamLote = tamLote;
	}

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
				"Esperado tensor com " + shapeEntrada.length +
				" ou " + (shapeEntrada.length + 1) +
				" dimensões. Recebido: " + x.numDim()
			);
		}

		_entrada.copiar(x);

		lops.forwardAvgPool2D(_entrada, _saida, _filtro, _stride);

		return _saida;
	}
	
	@Override
	public Tensor backward(Tensor g) {
		verificarConstrucao();

		if (g.numDim() != _entrada.numDim()) {
			throw new IllegalStateException(
				"\nEsperado gradiente " + _entrada.numDim() + "D, " +
				" mas recebido " + g.numDim() + "D."
			);
		}

		_gradEntrada.zero();// limpar acumulações anteriores

		lops.backwardAvgPool(_entrada, g, _gradEntrada, _filtro, _stride);

		return _gradEntrada;
	}

	@Override
	public Tensor saida() {
		verificarConstrucao();
		return _saida;
	}

	@Override
	public int[] shapeSaida() {
		verificarConstrucao();
		return shapeSaida;
	}

	@Override
	public int[] shapeEntrada() {
		verificarConstrucao();
		return shapeEntrada;
	}

	/**
	 * Retorna o formato do filtro (altura, largura) usado pela camada.
	 * @return formato do filtro da camada.
	 */
	public int[] formatoFiltro() {
		verificarConstrucao();
		return new int[]{
			_filtro[0],
			_filtro[1]
		};
	}

	/**
	 * Retorna o formato dos strides (altura, largura) usado pela camada.
	 * @return formato dos strides da camada.
	 */
	public int[] formatoStride() {
		verificarConstrucao();
		return new int[]{
			_stride[0],
			_stride[1]
		};
	}

	@Override
	public int numParams() {
		return 0;
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
	public AvgPool2D clone() {
		AvgPool2D clone = (AvgPool2D) super.clone();

		clone.lops = new LayerOps();
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
