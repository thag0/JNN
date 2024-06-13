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
	 * Utilitario.
	 */
	Utils utils = new Utils();

	/**
	 * Dimensões dos dados de entrada (canais, altura, largura)
	 */
	private final int[] shapeEntrada = {1, 1, 1};

	/**
	 * Dimensões dos dados de saída (canais, altura, largura)
	 */
	private final int[] shapeSaida = {1, 1, 1};

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
	private int[] formFiltro;

	/**
	 * Valores de stride (altura, largura).
	 */
	private int[] stride;

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
	 * @param formFiltro formato do filtro de average pooling.
	 */
	public AvgPool2D(int[] formFiltro) {
		utils.validarNaoNulo(formFiltro, "O formato do filtro não pode ser nulo.");

		if (formFiltro.length != 2) {
			throw new IllegalArgumentException(
				"\nO formato do filtro deve conter dois elementos (altura, largura)."
			);
		}

		if (!utils.apenasMaiorZero(formFiltro)) {
			throw new IllegalArgumentException(
				"\nOs valores de dimensões do filtro devem ser maiores que zero."
			);
		}

		this.formFiltro = formFiltro;
		this.stride = new int[]{
			formFiltro[0],
			formFiltro[1]
		};
	}

	/**
	 * Instancia uma nova camada de average pooling, definindo o formato do filtro 
	 * e os strides (passos) que serão aplicados em cada entrada da camada.
	 * <p>
	 *    O formato do filtro e dos strides devem conter as dimensões da entrada 
	 *    da camada (altura, largura).
	 * </p>
	 * @param formFiltro formato do filtro de average pooling.
	 * @param stride strides que serão aplicados ao filtro.
	 */
	public AvgPool2D(int[] formFiltro, int[] stride) {
		utils.validarNaoNulo(formFiltro, "O formato do filtro não pode ser nulo.");

		if (formFiltro.length != 2) {
			throw new IllegalArgumentException(
				"\nO formato do filtro deve conter dois elementos (altura, largura)."
			);
		}

		if (!utils.apenasMaiorZero(formFiltro)) {
			throw new IllegalArgumentException(
				"\nOs valores de dimensões do filtro devem ser maiores que zero."
			);
		}

		utils.validarNaoNulo(stride, "O formato dos strides não pode ser nulo.");

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

		this.formFiltro = formFiltro;
		this.stride = stride;
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
	 * @param formEntrada formato de entrada para a camada.
	 * @param formFiltro formato do filtro de average pooling.
	 * @param stride strides que serão aplicados ao filtro.
	 */
	public AvgPool2D(int[] formEntrada, int[] formFiltro, int[] stride) {
		this(formFiltro, stride);
		construir(formEntrada);
	}

	/**
	 * Constroi a camada AvgPooling, inicializando seus atributos.
	 * <p>
	 *    O formato de entrada da camada deve seguir o padrão:
	 * </p>
	 * <pre>
	 *    formEntrada = (canais, altura, largura)
	 * </pre>
	 * <h3>
	 *    Nota
	 * </h3>
	 * <p>
	 *    Caso o formato de entrada contenha quatro elementos, o primeiro
	 *    valor é descondiderado.
	 * </p>
	 * @param entrada formato dos dados de entrada para a camada.
	 */
	@Override
	public void construir(Object entrada) {
		utils.validarNaoNulo(entrada, "Formato de entrada fornecido para camada " + nome() + " é nulo.");

		if (!(entrada instanceof int[])) {
			throw new IllegalArgumentException(
				"\nObjeto esperado para entrada da camada " + nome() + " é do tipo int[], " +
				"objeto recebido é do tipo " + entrada.getClass().getTypeName()
			);
		}
		
		int[] e = (int[]) entrada;
		if (e.length != 3) {
			throw new IllegalArgumentException(
				"\nFormato de entrada para a camada " + nome() + " deve conter três " + 
				"elementos (canais, altura, largura), mas recebido tamanho = " + e.length
			);
		}

		shapeEntrada[0] = e[0];
		shapeEntrada[1] = e[1];
		shapeEntrada[2] = e[2];

		shapeSaida[0] = shapeEntrada[0];// canais
		shapeSaida[1] = (shapeEntrada[1] - formFiltro[0]) / this.stride[0] + 1;//altura
		shapeSaida[2] = (shapeEntrada[2] - formFiltro[1]) / this.stride[1] + 1;//largura
		
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
			Tensor e = (Tensor) entrada;

			if (!(_entrada.compararShape(e))) {
				throw new IllegalArgumentException(
					"\nDimensões da entrada recebida " + e.shapeStr() +
					" incompatíveis com a entrada da camada " + this._entrada.shapeStr()
				);
			}

			_entrada.copiar(e);
			
		} else if (entrada instanceof double[][][]) {
			double[][][] e = (double[][][]) entrada;
			_entrada.copiar(e);

		} else {
			throw new IllegalArgumentException(
				"\nTipo de entrada \"" + entrada.getClass().getTypeName() + "\" não suportada."
			);
		}

		int canais = shapeEntrada[0];
		for (int i = 0; i < canais; i++) {
			aplicar(_entrada, _saida, i);
		}

		return _saida;
	}

	/**
	 * Calcula a média dos valores encontrados na entrada de acordo com as
	 * configurações de filtro e strides.
	 * @param entrada tensor de entrada.
	 * @param saida tensor de destino.
	 * @param prof índice de profundidade da operação.
	 */
	private void aplicar(Tensor entrada, Tensor saida, int prof) {
		int[] shapeE = entrada.shape();
		int[] shapeS = saida.shape();

		int altEntrada  = shapeE[1];
		int largEntrada = shapeE[2];
		int altSaida  = shapeS[1];
		int largSaida = shapeS[2];
  
		for (int i = 0; i < altSaida; i++) {
			int linInicio = i * stride[0];
			int linFim = Math.min(linInicio + formFiltro[0], altEntrada);
			for (int j = 0; j < largSaida; j++) {
				int colInicio = j * stride[1];
				int colFim = Math.min(colInicio + formFiltro[1], largEntrada);
				double soma = 0;
				int cont = 0;

				for (int lin = linInicio; lin < linFim; lin++) {
					for (int col = colInicio; col < colFim; col++) {
						soma += entrada.get(prof, lin, col);
						cont++;
					}
				}

				saida.set((soma/cont), prof, i, j);
			}
		}
	}

	@Override
	public Tensor backward(Object grad) {
		verificarConstrucao();

		if (grad instanceof Tensor) {
			Tensor g = (Tensor) grad;
			int canais = shapeEntrada[0];   
			for (int i = 0; i < canais; i++) {
				gradAvgPool(_entrada, g, _gradEntrada, i);
			}
		
		} else {
			throw new IllegalArgumentException(
				"Formato de gradiente \" "+ grad.getClass().getTypeName() +" \" não " +
				"suportado para camada de AvgPooling."
			);
		}

		return _gradEntrada;
	}

	/**
	 * Calcula e atualiza os gradientes da camada de Avg Pooling em relação à entrada.
	 * <p>
	 *    Retroropaga os gradientes da camada seguinte para a camada de Avg Pooling, considerando 
	 *    a operação de agrupamento médio. Ela calcula os gradientes em relação à entrada para as 
	 *    camadas anteriores.
	 * </p>
	 * @param entrada entrada da camada.
	 * @param gradSeguinte gradiente da camada seguinte.
	 * @param gradEntrada gradiente de entrada da camada de Avg pooling.
	 * @param prof índice de profundidade da operação.
	 */
	private void gradAvgPool(Tensor entrada, Tensor gradSeguinte, Tensor gradEntrada, int prof) {
		int[] shapeE = entrada.shape();
		int[] shapeGradS = gradSeguinte.shape();

		int altEntrada  = shapeE[shapeE.length-1];
		int largEntrada = shapeE[shapeE.length-2];
		int altGradSeguinte  = shapeGradS[shapeGradS.length-1];
		int largGradSeguinte = shapeGradS[shapeGradS.length-2];

		for (int i = 0; i < altGradSeguinte; i++) {
			int linInicio = i * stride[0];
			int linFim = Math.min(linInicio + formFiltro[0], altEntrada);
			for (int j = 0; j < largGradSeguinte; j++) {
				int colInicio = j * stride[1];
				int colFim = Math.min(colInicio + formFiltro[1], largEntrada);

				double grad = gradSeguinte.get(prof, i, j);
				double mediaGrad = grad / (formFiltro[0] * formFiltro[1]);

				for (int lin = linInicio; lin < linFim; lin++) {
					for (int col = colInicio; col < colFim; col++) {
						gradEntrada.set(mediaGrad, prof, lin, col);
					}
				}
			}
		}
	}

	@Override
	public Tensor saida() {
		verificarConstrucao();
		return _saida;
	}

	@Override
	public int[] formatoSaida() {
		verificarConstrucao();
		return shapeSaida;
	}

	@Override
	public int[] formatoEntrada() {
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
			formFiltro[0],
			formFiltro[1]
		};
	}

	/**
	 * Retorna o formato dos strides (altura, largura) usado pela camada.
	 * @return formato dos strides da camada.
	 */
	public int[] formatoStride() {
		verificarConstrucao();
		return new int[]{
			stride[0],
			stride[1]
		};
	}

	@Override
	public int numParametros() {
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
		sb.append(pad).append("Filtro: " + utils.shapeStr(formFiltro) + "\n");
		sb.append(pad).append("Strides: " + utils.shapeStr(stride) + "\n");
		sb.append(pad).append("Saída: " + utils.shapeStr(formatoSaida()) + "\n");

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
	
}
