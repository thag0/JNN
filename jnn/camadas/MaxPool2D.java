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
	 * @param formFiltro formato do filtro de max pooling.
	 * @throws IllegalArgumentException se o formato do filtro não atender as
	 * requisições.
	 */
	public MaxPool2D(int[] formFiltro) {
		this(formFiltro, formFiltro.clone());
	}

	/**
	 * Instancia uma nova camada de max pooling, definindo o formato do filtro 
	 * e os strides (passos) que serão aplicados em cada entrada da camada.
	 * <p>
	 *    O formato do filtro e dos strides devem conter as dimensões da entrada 
	 *    da camada (altura, largura).
	 * </p>
	 * @param formFiltro formato do filtro de max pooling.
	 * @param stride strides que serão aplicados ao filtro.
	 * @throws IllegalArgumentException se o formato do filtro não atender as
	 * requisições.
	 * @throws IllegalArgumentException se os strides não atenderem as requisições.
	 */
	public MaxPool2D(int[] formFiltro, int[] stride) {
		utils.validarNaoNulo(formFiltro, "\nO formato do filtro não pode ser nulo.");

		if (formFiltro.length != 2) {
			throw new IllegalArgumentException(
				"\nO formato do filtro deve conter três elementos (altura, largura)."
			);
		}

		if (!utils.apenasMaiorZero(formFiltro)) {
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

		this._filtro = formFiltro.clone();
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
	 * @param formEntrada formato de entrada para a camada.
	 * @param formFiltro formato do filtro de max pooling.
	 * @param stride strides que serão aplicados ao filtro.
	 * @throws IllegalArgumentException se o formato do filtro não atender as
	 * requisições.
	 * @throws IllegalArgumentException se os strides não atenderem as requisições.
	 */
	public MaxPool2D(int[] formEntrada, int[] formFiltro, int[] stride) {
		this(formFiltro, stride);
		construir(formEntrada);
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
		
		_entrada = new Tensor(shapeEntrada);
		_gradEntrada = new Tensor(_entrada);
		_saida = new Tensor(shapeSaida);

		setNomes();

		_construida = true;//camada pode ser usada
	}

	@Override
	public void inicializar() {}

	@Override
	protected void setNomes() {
		_entrada.nome("entrada");
		_gradEntrada.nome("gradiente entrada");
		_saida.nome("saída");
	}

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

		optensor.maxPool2D(_entrada, _saida, _filtro, _stride);

		return _saida;
	}

	@Override
	public Tensor backward(Object grad) {
		verificarConstrucao();

		if (grad instanceof Tensor) {
			Tensor g = (Tensor) grad;
			int profundidade = shapeEntrada[0];   
			for (int i = 0; i < profundidade; i++) {
				gradMaxPool(_entrada, g, _gradEntrada, i);
			}
		
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
	 * @param prof índice de profundidade da operação.
	 */
	private void gradMaxPool(Tensor entrada, Tensor gradSeguinte, Tensor gradEntrada, int prof) {
		int[] shapeEntrada = entrada.shape();
		int[] shapeGradS = gradSeguinte.shape();

		int altEntrada  = shapeEntrada[1];
		int largEntrada = shapeEntrada[2];

		int altGradSeguinte  = shapeGradS[1];
		int largGradSeguinte = shapeGradS[2];
  
		for (int i = 0; i < altGradSeguinte; i++) {
			int linInicio = i * _stride[0];
			int linFim = Math.min(linInicio + _filtro[0], altEntrada);
			for (int j = 0; j < largGradSeguinte; j++) {
				int colInicio = j * _stride[1];
				int colFim = Math.min(colInicio + _filtro[1], largEntrada);
  
				int[] posicaoMaximo = posicaoMaxima(entrada, prof, linInicio, colInicio, linFim, colFim);
				int linMaximo = posicaoMaximo[0];
				int colMaximo = posicaoMaximo[1];
  
				gradEntrada.set(
					gradSeguinte.get(prof, i, j), 
					prof, linMaximo, colMaximo
				);
			}
		}
	}

	/**
	 * Encontra a posição do valor máximo em uma submatriz do tensor.
	 * <p>
	 *    Se houver múltiplos elementos com o valor máximo, a função retorna as coordenadas 
	 *    do primeiro encontrado.
	 * </p>
	 * @param tensor tensor alvo.
	 * @param linInicio índice inicial para linha.
	 * @param colInicio índice final para a linha.
	 * @param linFim índice inicial para coluna (exclusivo).
	 * @param colFim índice final para coluna (exclusivo).
	 * @param prof índice de profundidade da operação.
	 * @return array representando as coordenadas (linha, coluna) do valor máximo
	 * na submatriz.
	 */
	private int[] posicaoMaxima(Tensor tensor, int prof, int linInicio, int colInicio, int linFim, int colFim) {
		int[] posMaximo = {0, 0};
		double valMaximo = Double.NEGATIVE_INFINITY;
  
		for (int i = linInicio; i < linFim; i++) {
			for (int j = colInicio; j < colFim; j++) {
				if (tensor.get(prof, i, j) > valMaximo) {
					valMaximo = tensor.get(prof, i, j);
					posMaximo[0] = i;
					posMaximo[1] = j;
				}
			}
		}
  
		return posMaximo;
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
	
	@Override
	public void copiarParaTreinoLote(Camada camada) {
		if (!(camada instanceof MaxPool2D)) {
			throw new UnsupportedOperationException(
				"\nCamada deve ser do tipo " + getClass() +
				", mas é do tipo " + camada.getClass()
			);
		}

		MaxPool2D c = (MaxPool2D) camada;
		_entrada.copiar(c._entrada);
		_saida.copiar(c._saida);
	}

}
