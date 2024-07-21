package jnn.camadas;

import jnn.core.Utils;
import jnn.core.tensor.Tensor;
import jnn.core.tensor.Variavel;

/**
 * <h2>
 *    Camada flatten ou "achatadora"
 * </h2>
 * <p>
 *    Transforma os recebidos no formato sequencial.
 * </p>
 * <p>
 *    A camada de flatten não possui parâmetros treináveis nem função de ativação.
 * </p>
 */
public class Flatten extends Camada implements Cloneable{

	/**
	 * Utilitário.
	 */
	Utils utils = new Utils();

	/**
	 * Array contendo o formato de entrada da camada, de acordo com o formato:
	 * <pre>
	 *    entrada = (..., profundidade, altura, largura)
	 * </pre>
	 */
	private int[] shapeEntrada;

	/**
	 * Array contendo o formato de saida da camada, de acordo com o formato:
	 * <pre>
	 *    saida = (elementosTotaisEntrada)
	 * </pre>
	 */
	private int[] shapeSaida;

	/**
	 * Tensor contendo os valores de entrada para a camada,
	 * que serão usados para o processo de feedforward.
	 * <p>
	 *    O formato da entrada é dado por:
	 * </p>
	 * <pre>
	 *    entrada = (..., profundidade, altura, largura)
	 * </pre>
	 */
	public Tensor _entrada;
	
	/**
	 * Tensor contendo os valores dos gradientes usados para 
	 * a retropropagação para camadas anteriores.
	 * <p>
	 *    O formato dos gradientes é dado por:
	 * </p>
	 * <pre>
	 *    gradEntrada = (..., profundidadeEntrada, alturaEntrada, larguraEntrada)
	 * </pre>
	 */
	public Tensor _gradEntrada;

	/**
	 * Tensor contendo a saída achatada da camada.
	 * <p>
	 *    Mesmo a saída sendo um tensor, ela possui apenas
	 *    uma linha e o número de colunas é equivalente a quantidade
	 *    total de elementos da entrada.
	 * </p>
	 */
	public Tensor _saida;

	/**
	 * Inicializa uma camada Flatten, que irá achatar a entrada recebida
	 * no formato de um tensor unidimensional.
	 * <p>
	 *    É necessário construir a camada para que ela possa ser usada.
	 * </p>
	 */
	public Flatten() {}

	/**
	 * Instancia uma camada Flatten, que irá achatar a entrada recebida
	 * no formato de um tensor unidimensional.
	 * <p>
	 *    O formato de entrada da camada deve seguir o formato:
	 * </p>
	 * <pre>
	 *    formEntrada = (profundidade, altura, largura)
	 * </pre>
	 * @param formEntrada formato dos dados de entrada para a camada.
	 */
	public Flatten(int[] formEntrada) {
		construir(formEntrada);
	}

	/**
	 * Inicializa os parâmetros necessários para a camada Flatten.
	 * <p>
	 *    O formato de entrada deve ser um array contendo o tamanho de 
	 *    cada dimensão de entrada da camada, e deve estar no formato:
	 * </p>
	 * <pre>
	 *    entrada = (profundidade, altura, largura)
	 * </pre>
	 * Também pode ser aceito um objeto de entrada contendo apenas dois elementos,
	 * eles serão formatados como:
	 * <pre>
	 *    entrada = ( altura, largura)
	 * </pre>
	 */
	@Override
	public void construir(int[] shape) {
		utils.validarNaoNulo(shape, "Formato de entrada nulo.");

		if (!utils.apenasMaiorZero(shape)) {
			throw new IllegalArgumentException(
				"\nOs valores do formato de entrada devem ser maiores que zero."
			);
		}

		shapeEntrada = shape.clone();

		int tamanho = 1;
		for (int i : shapeEntrada) {
			tamanho *= i;
		}

		shapeSaida = new int[]{tamanho};

		_entrada = new Tensor(shapeEntrada);
		_gradEntrada = new Tensor(_entrada.shape());
		_saida = new Tensor(shapeSaida);

		setNomes();

		_construida = true;// camada pode ser usada.
	}

	@Override
	public void inicializar() {}

	@Override
	public void setSeed(long seed) {}

	@Override
	protected void setNomes() {
		_entrada.nome("entrada");
		_saida.nome("saída");
		_gradEntrada.nome("gradiente entrada");     
	}

	/**
	 * <h2>
	 *    Propagação direta através da camada Flatten
	 * </h2>
	 * Achata os dados de entrada num formato sequencial.
	 * @param entrada dados de entrada que serão processados, objetos aceitos incluem:
	 * {@code Tensor}
	 */
	@Override
	public Tensor forward(Object entrada) {
		verificarConstrucao();

		if (entrada instanceof Tensor) {
			_entrada.copiar((Tensor) entrada);

		} else {
			throw new IllegalArgumentException(
				"\nTipo de entrada \"" + entrada.getClass().getTypeName() + "\"" +
				" não suportada."
			);
		}

		_saida.copiarElementos(_entrada);

		return _saida;
	}

	/**
	 * <h2>
	 *    Propagação reversa através da camada Flatten
	 * </h2>
	 * Desserializa os gradientes recebedos de volta para o mesmo formato de entrada.
	 * @param grad gradientes de entrada da camada seguinte, objetos aceitos incluem:
	 * {@code Tensor}.
	 */
	@Override
	public Tensor backward(Object grad) {
		verificarConstrucao();

		if (grad instanceof Tensor) {
			_gradEntrada.copiarElementos((Tensor) grad);
		
		} else {
			throw new IllegalArgumentException(
				"\nTipo de gradiente \"" + grad.getClass().getTypeName() + "\"" +
				" não suportado."
			);
		}

		return _gradEntrada;
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
	public int tamanhoSaida() {
		return saida().tamanho();
	}

	@Override
	public int[] shapeEntrada() {
		verificarConstrucao();
		return shapeEntrada.clone();
	}

	/**
	 * Calcula o formato de saída da camada Flatten, que é disposto da
	 * seguinte forma:
	 * <pre>
	 *    formato = (elementosEntrada)
	 * </pre>
	 * Onde {@code elementosEntrada} é a quantidade total de elementos 
	 * contidos no formato de entrada da camada.
	 * @return formato de saída da camada
	 */
	 @Override
	public int[] shapeSaida() {
		verificarConstrucao();
		
		return new int[]{
			tamanhoSaida()
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
		
		sb.append(nome() + " (id " + id + ") = [\n");

		sb.append(pad).append("Entrada: " + utils.shapeStr(shapeEntrada) + "\n");
		sb.append(pad).append("Saída: (1, " + tamanhoSaida() + ")\n");

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
	public Flatten clone() {
		Flatten clone = (Flatten) super.clone();

		clone._treinavel = this._treinavel;
		clone.treinando = this.treinando;
		clone._construida = this._construida;

		clone.shapeEntrada = shapeEntrada.clone();
		clone.shapeSaida = shapeSaida.clone();

		clone._entrada = _entrada.clone();
		clone._gradEntrada = _gradEntrada.clone();
		clone._saida = _saida.clone();

		return clone;
	}

}
